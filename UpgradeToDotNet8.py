import os
import re
import openai
import logging
import concurrent.futures

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Configuration
CHUNK_SIZE = 200_000  # Max chunk size in characters
AZURE_DEPLOYMENT_NAME = "gpt-4o"
ROOT_PATH = "<path>"

# Enhanced System Prompt
SYSTEM_PROMPT = """You are an assistant that helps upgrade older C# code to be compatible with .NET 8.

Your task is to:
- Identify and update any code patterns, APIs, or configurations that are incompatible with .NET 8, even if they originate from versions earlier than .NET 4.8.
- If the code is outdated but still compatible with .NET 8, leave it unchanged unless there's a strong reason to modernize it.
- If the existing code is not incompatible with .NET 8, return 'compatible'.
- Ensure any code that must be changed compiles and follows best practices for .NET 8.
- Leave all code that is not incompatible with .NET 8 alone, including comments. Simplifying whitespace is fine.

If you encounter legacy APIs or patterns that have a clear modern equivalent (e.g., Web Forms, System.Web, or FormsAuth), implement it where feasible.
"""

USER_PROMPT_TEMPLATE = """Please upgrade the following .NET 4.8 C# code to .NET 8.
If it is already compatible, return 'compatible'.

Chunk context (from entire file):
{chunk_context}

Code:
```csharp
{code_chunk}
```
"""


def get_csharp_files(base_path):
    """
    Recursively collect all .cs files from the given base path.
    """
    csharp_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".cs"):
                csharp_files.append(os.path.join(root, f))
    return csharp_files


def extract_context(csharp_code):
    """
    Extract top-level context: using statements, namespace, class/struct/record names, and method signatures.
    """
    using_statements = re.findall(r"^\s*using\s+[\w\.]+;", csharp_code, re.MULTILINE)
    class_names = re.findall(
        r"(?:public|internal|private|protected)\s+(?:partial\s+)?(class|struct|record)\s+(\w+)",
        csharp_code,
    )
    method_signatures = re.findall(
        r"(?:public|internal|private|protected)\s+(?:override\s+)?(?:virtual\s+)?\w[\w<>]*\s+(\w+)\(.*?\)",
        csharp_code,
    )
    namespace_match = re.search(r"^\s*namespace\s+([\w\.]+)", csharp_code, re.MULTILINE)

    context_parts = []

    if using_statements:
        context_parts.append("\n".join(using_statements))
    if namespace_match:
        context_parts.append(f"Namespace: {namespace_match.group(1)}")
    if class_names:
        classes_str = ", ".join([f"{c[0]} {c[1]}" for c in class_names])
        context_parts.append(f"Classes: {classes_str}")
    if method_signatures:
        method_signatures_str = ", ".join(method_signatures)
        context_parts.append(f"Methods: {method_signatures_str}")

    return "\n".join(context_parts).strip()


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Split text into chunks of a specified size.
    This is a simple approach; you could enhance it to split by method boundaries or other logical breaks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks


def upgrade_code_chunk(code_chunk, chunk_context):
    """
    Call Azure OpenAI API for a single code chunk, returning the model's response or the original chunk on error.
    """
    combined_code = chunk_context + "\n\n" + code_chunk if chunk_context else code_chunk

    user_prompt = USER_PROMPT_TEMPLATE.format(
        chunk_context=chunk_context, code_chunk=combined_code
    )

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        updated_content = response["choices"][0]["message"]["content"]
        return updated_content
    except Exception as e:
        logging.error(f"Error calling Azure OpenAI API: {e}")
        return code_chunk


def process_file(cs_file):
    """
    Process a single .cs file: chunking, context extraction, calling the upgrade API,
    and writing the updated file back to disk.
    """
    logging.info(f"Processing: {cs_file}")
    with open(cs_file, "r", encoding="utf-8") as f:
        original_code = f.read()

    chunks = chunk_text(original_code, CHUNK_SIZE)

    context = extract_context(original_code) if len(chunks) > 1 else ""

    updated_file_content = []
    for chunk in chunks:
        updated_chunk = upgrade_code_chunk(chunk, context)
        if updated_chunk.strip() == "compatible":
            updated_chunk = chunk
        updated_file_content.append(updated_chunk)

    final_code = "\n".join(updated_file_content)

    with open(cs_file, "w", encoding="utf-8") as out_f:
        out_f.write(final_code)

    logging.info(f"Finished processing: {cs_file}")


def main():
    """
    Upgrade all .cs files in the input directory, using parallel processing to speed things up.
    """
    cs_files = get_csharp_files(ROOT_PATH)
    if not cs_files:
        logging.warning("No .cs files found in the specified directory.")
        return

    logging.info(f"Found {len(cs_files)} .cs files. Starting upgrade...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, cs_file) for cs_file in cs_files]
        concurrent.futures.wait(futures)

    logging.info("All files processed. Upgrade complete.")


if __name__ == "__main__":
    main()
