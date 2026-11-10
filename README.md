# Address Extraction with Qwen 3.5

Turn messy address text into structured street address, city, state, ZIP code, and explanation fields using **Qwen 3.5-9B**, **Google Colab**, and your own examples.

This notebook processes a CSV one row at a time, keeps the original text alongside the extracted fields, records processing errors and elapsed time, and exports the results as a CSV you can review in Excel or use in a later data workflow.

The model runs inside the Colab runtime. Address extraction does not use a paid language model API. Colab is a cloud environment, so uploaded data is still processed on Google's infrastructure.

> **Scope:** This is an address extraction and cleaning workflow for US street addresses. It does not verify deliverability, query an authoritative address database, or produce latitude and longitude. Review the results before using them for geocoding or other downstream work.

## Contents

- [What the notebook does](#what-the-notebook-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Repository files](#repository-files)
- [Quick start](#quick-start)
- [Prepare your CSV files](#prepare-your-csv-files)
- [Settings](#settings)
- [Notebook sections explained](#notebook-sections-explained)
- [Output file and columns](#output-file-and-columns)
- [Error handling and saved progress](#error-handling-and-saved-progress)
- [Performance and memory](#performance-and-memory)
- [Reviewing quality](#reviewing-quality)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Data handling and publishing](#data-handling-and-publishing)
- [Reproducibility](#reproducibility)
- [License](#license)

## What the notebook does

The workflow asks the model to:

- Extract the primary street number, street name, and directional information.
- Identify the city, two-letter US state abbreviation, and five-digit ZIP code.
- Remove recipient names, businesses, departments, phone numbers, email addresses, reference numbers, HTML, and delivery instructions.
- Remove apartment, suite, unit, floor, room, and secondary building identifiers for street-level geocoding.
- Prefer an address explicitly marked `CURRENT`, `NEW`, or `PHYSICAL` over an old or mailing address.
- Avoid inventing information that the source text does not support.
- Leave the street address empty when there is no usable physical street address or no clear choice between conflicting addresses.
- Keep recoverable city, state, and ZIP information even when the street address is unavailable.
- Provide a short explanation in the `rationale` field.

These are instructions to a language model, so they describe intended behavior rather than guaranteed validation rules.

## How it works

Each extraction request contains three parts:

1. **System instructions:** The address extraction rules and required JSON fields.
2. **Your examples:** Alternating raw input and expected-output pairs built from the examples CSV.
3. **The current record:** One `raw_text` value from the input CSV.

This is **few-shot prompting**. The examples demonstrate the desired behavior inside each prompt. They do not train or fine-tune the model, update its weights, or create a new model file.

All uploaded example rows are included in every extraction request. Input rows are processed separately, rather than being combined into one large request. The model is asked to return JSON, which the notebook parses into CSV columns.

## Requirements

### Runtime

- Google Colab with a GPU runtime enabled.
- A Google account with access to Colab.
- Internet access for package installation and the initial model download.
- Enough GPU memory, system memory, and disk space for the model and prompts.
- Two CSV files: one containing examples and one containing records to clean.

The notebook checks for CUDA and stops if a GPU is unavailable. It does not implement a CPU-only mode. It includes handling for optional optimization availability on GPUs such as the T4, but it does not guarantee that every hardware and package combination will work.

### Python dependencies

The installation cell requests these packages:

| Package | Requested version | Purpose |
| --- | --- | --- |
| `pandas` | `==2.2.3` | Read, validate, preview, and export CSV data. |
| `transformers` | `>=5.0,<6` | Load the tokenizer and Qwen model; generate responses. |
| `accelerate` | `>=1.4,<2` | Support model loading and automatic device placement. |
| `bitsandbytes` | `>=0.46,<1` | Load model weights with 4-bit quantization. |
| `safetensors` | Unpinned | Support model weight loading. |
| `sentencepiece` | Unpinned | Tokenization dependency. |
| `einops` | Unpinned | Tensor manipulation dependency. |

The notebook keeps Colab's existing PyTorch installation. It also uses `tqdm.auto`, `google.colab.files`, and the notebook's `display()` support. Those are expected from the Colab environment; `tqdm` is not explicitly listed in the installation cell.

Optional packages are `causal-conv1d` and `flash-linear-attention`. The notebook attempts binary-wheel installation only, without installing their dependencies. It does not compile them from source.

## Repository files

| File | Purpose |
| --- | --- |
| `README.md` | Setup, usage, implementation details, and troubleshooting. |
| `qwen_address_extraction_colab(1).ipynb` | The supplied Google Colab notebook. |

The notebook opens upload dialogs for your CSV files. No particular CSV filename is required, and this README does not assume sample datasets are included in the repository.

You can rename the notebook before publishing. If you do, update its filename in this README to match.

## Quick start

1. Open [Google Colab](https://colab.research.google.com/).
2. Use the notebook picker to open the notebook from GitHub, or download the `.ipynb` file and upload it to Colab.
3. Select **Runtime → Change runtime type** and choose a GPU hardware accelerator.
4. Run the notebook cells from top to bottom.
5. Allow the dependency installation to finish.
6. Run the GPU check and optional optimization cell.
7. At the first upload prompt, choose **exactly one examples CSV**.
8. At the second upload prompt, choose **exactly one input CSV**.
9. Load the model. The initial model download and loading take time.
10. Run the extraction rules cell. It automatically tests the first input row and displays its result.
11. Review that test before running the full extraction cell.
12. For an initial trial, set `MAX_ROWS` to a positive integer such as `5` or `10` in the extraction cell. Use `None` to process all rows.
13. Run extraction, then run the download cell to save the cleaned CSV to your computer.

The first-row test is separate from the main extraction loop. The loop processes that row again, so the test does not count as a completed output row.

## Prepare your CSV files

Both uploads use a header row. **Column names are case-sensitive and must match exactly**, without extra spaces. For example, `raw_text` works; `Raw_Text` does not.

Both files are read with `dtype=str`, `keep_default_na=False`, and `encoding="utf-8-sig"`. This keeps values as text, preserves leading zeros in ZIP codes when they exist in the CSV, and keeps empty cells as empty strings instead of converting them to missing-value markers. UTF-8 files with or without a byte-order mark are supported by that encoding.

If a value contains a comma, quote the entire field. Spreadsheet software generally handles CSV quoting when exporting. Saving a file as CSV does not recover leading zeros that Excel has already removed.

### 1. Examples CSV

This file tells the model how you want messy text handled.

| Column | Required | Meaning |
| --- | --- | --- |
| `raw_text` | Yes | Messy source text. Every example must have a nonblank value. |
| `address` | Yes | Desired primary street address, without secondary unit details. |
| `city` | Yes | Desired city. |
| `state` | Yes | Desired two-letter US state abbreviation. |
| `zip` | Yes | Desired five-digit ZIP code, stored as text. |
| `rationale` | Yes | Brief explanation of the desired extraction or omission. |

All six headers must exist. The examples file must contain at least one data row. Output values can be blank when the source does not support them. Additional columns are not included in the model's example responses.

Use examples covering the actual problems in your data: extra names, unit numbers, old versus current addresses, incomplete addresses, and ambiguous records. Keep the examples consistent with the system instructions.

The following is an illustrative synthetic example, not a verified address or a claim about model accuracy:

| raw_text | address | city | state | zip | rationale |
| --- | --- | --- | --- | --- | --- |
| `ATTN Jane Example; 123 Example St Apt 4; Sampletown MI 00000` | `123 Example St` | `Sampletown` | `MI` | `00000` | `Removed recipient and apartment number.` |
| `No street provided; Sampletown MI 00000` | *(empty)* | `Sampletown` | `MI` | `00000` | `No physical street address provided.` |

Write an actual empty cell wherever the table says `empty`; do not enter the word itself.

### 2. Input CSV

This file contains the records to clean.

| Column | Required | Meaning |
| --- | --- | --- |
| `raw_text` | Yes | One messy address string per row. |
| `id` | No | A unique record identifier. |

If `id` is absent, the notebook creates sequential text IDs starting at `1`. If it is present, its values must be unique. Duplicate IDs stop input validation. A single blank ID is not explicitly rejected, so providing a nonblank ID for every row is recommended.

The input file must contain at least one data row. Unlike the examples validation, the input validation does not reject blank `raw_text` values; they are passed to the model.

Additional input columns may be present, but **they are not copied into the cleaned CSV**. Keep your original file and use `id` to join results back to it. If a column is important for later joins, include an explicit unique `id` before uploading.

## Settings

Settings are located in the notebook sections shown below; there is no separate configuration file.

| Setting | Default | Location | Effect |
| --- | --- | --- | --- |
| `TRY_PREBUILT_OPTIMIZATIONS` | `True` | GPU and optional optimizations | Attempts installation of missing optional binary wheels. Set to `False` to skip those installation attempts. |
| `MODEL_ID` | `"Qwen/Qwen3.5-9B"` | Load model | Hugging Face model identifier used for the tokenizer and model. |
| `MAX_NEW_TOKENS` | `160` | Load model | Maximum newly generated tokens for each response, excluding the prompt. |
| `FIELDS` | `address`, `city`, `state`, `zip`, `rationale` | Extraction rules and examples | Defines the expected example and result fields. |
| `SYSTEM_PROMPT` | Address extraction rules | Extraction rules and examples | Defines the intended cleaning behavior and output format. |
| `MAX_ROWS` | `None` | Extract addresses | Processes every row when `None`; otherwise uses the first specified number of rows. |

Use a **positive integer** for a subset test. Zero or negative values are not validated as sensible test settings; zero can leave the final summary without the expected columns.

Changing `MODEL_ID` alone does not make the notebook compatible with every model. It explicitly loads `Qwen3_5ForCausalLM`, and uses a chat template with `enable_thinking=False`. Other model families can require code changes.

### Model loading configuration

| Parameter | Value | Meaning |
| --- | --- | --- |
| `load_in_4bit` | `True` | Uses 4-bit weight loading through bitsandbytes. |
| `bnb_4bit_quant_type` | `"nf4"` | Selects NF4 quantization. |
| `bnb_4bit_compute_dtype` | `torch.float16` | Uses FP16 computation for quantized layers. |
| `bnb_4bit_use_double_quant` | `True` | Enables double quantization. |
| `device_map` | `"auto"` | Lets the loading libraries determine device placement. |
| `dtype` | `torch.float16` | Requests FP16 for applicable model components. |
| `low_cpu_mem_usage` | `True` | Requests a memory-conscious loading path. |

This is a Transformers/bitsandbytes workflow, not a GGUF or llama.cpp workflow. Quantization reduces weight memory, but does not eliminate prompt, cache, loading, or runtime memory requirements.

### Generation configuration

The model runs with `model.eval()` and `torch.inference_mode()`. Generation uses `do_sample=False`, so random token sampling is disabled. The tokenizer adds the chat generation prompt, returns a dictionary of tensors, and receives `enable_thinking=False`.

Only newly generated tokens are decoded. This prevents the prompt and examples from being included in the parsed response.

## Notebook sections explained

### 1. Install dependencies

Installs the listed Python packages into the active runtime. PyTorch is left in place. If package changes produce import or binary compatibility errors, restarting the Colab session and rerunning the cells may be necessary.

### 2. GPU and optional optimizations

Checks CUDA availability and displays the GPU name, compute capability, PyTorch version, and CUDA version reported by PyTorch.

For each optional optimization, the cell checks whether its module is discoverable. If missing and installation attempts are enabled, it tries a prebuilt binary wheel. It then reports whether the module is missing, imports successfully, or fails to import.

An import success message does not verify that a kernel supports the current GPU. Missing optional packages can allow a slower PyTorch path; an installed but incompatible extension may still cause errors during inference.

### 3. Upload example CSV

Accepts exactly one file, checks the six required headers, rejects an empty dataset or blank example inputs, and previews the first rows.

### 4. Upload input CSV

Accepts exactly one file, checks `raw_text`, rejects an empty dataset, creates IDs if necessary, checks ID uniqueness, and previews the first rows.

### 5. Load model

Loads the tokenizer and the quantized Qwen model, switches to evaluation mode, and prints allocated GPU memory. That printed allocation is a snapshot, not a measurement of peak memory or total runtime resource use.

### 6. Extraction rules and examples

This section connects the uploaded examples to the model. It defines the extraction instructions, converts examples into chat messages, defines two functions, and runs a first-row test.

Defining a function makes it available for later calls; it does not execute its body immediately. The example-building loop runs when this cell executes. The model runs only when the cell reaches the call to `extract_address()` near the bottom.

#### Complete Section 6 code

The following is the complete code from this notebook section, reproduced for reference. It relies on the imports, uploaded DataFrames, tokenizer, model, and settings created by the preceding notebook cells; it is not a standalone script.

```python
import json
import re
import time
import torch

FIELDS = ("address", "city", "state", "zip", "rationale")

SYSTEM_PROMPT = """
Extract a physical street address for geocoding from one messy text string.

Return exactly one JSON object with these five string fields:
address, city, state, zip, rationale.

RULES:
1. Keep the primary street number, street name, and directional.
2. Extract the city, two-letter US state abbreviation, and five-digit ZIP.
3. Remove recipient names, ATTN lines, care-of information, businesses,
   departments, phone numbers, emails, invoices, reference numbers,
   websites, HTML, delivery instructions, and unnecessary punctuation.
4. Remove apartment, suite, unit, floor, room, and secondary building
   identifiers that are not needed for street-level geocoding.
5. Prefer addresses explicitly labeled CURRENT, NEW, or PHYSICAL
   over OLD addresses or mailing addresses.
6. Correct obvious typos only when supported by the original text.
7. Never invent unsupported street numbers, streets, cities, or ZIPs.
8. If no usable physical street address exists, leave address empty.
9. If multiple conflicting addresses exist without a clear choice,
   leave address empty.
10. Preserve city, state, and ZIP when recoverable, even if the
    street address cannot be determined.
11. Explain important changes or missing information in one
    short rationale.

Return valid JSON only. No markdown or additional commentary.
"""

# Build few-shot examples from the uploaded CSV.
demonstrations = []

for _, row in examples_df.iterrows():
    expected = {
        field: str(row[field])
        for field in FIELDS
    }

    demonstrations.extend([
        {
            "role": "user",
            "content": "Raw address: " + str(row["raw_text"])
        },
        {
            "role": "assistant",
            "content": json.dumps(expected, ensure_ascii=False)
        }
    ])


def parse_response(text):
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    ).strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*?\}", text)

        if not match:
            raise ValueError(
                "No JSON found in model response: " + text[:200]
            )

        result = json.loads(match.group())

    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object.")

    cleaned = {
        field: str(result.get(field, "") or "").strip()
        for field in FIELDS
    }

    # Standardize state capitalization.
    if cleaned["state"]:
        cleaned["state"] = cleaned["state"].upper()

    # Convert ZIP+4 to five digits.
    if re.fullmatch(r"\d{5}-\d{4}", cleaned["zip"]):
        cleaned["zip"] = cleaned["zip"][:5]

    return cleaned


def extract_address(raw_text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(demonstrations)

    messages.append({
        "role": "user",
        "content": "Raw address: " + str(raw_text)
    })

    # FIX: Return a dictionary of model inputs.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False
    )

    # Move tokenized inputs to the model device.
    inputs = inputs.to(model.device)

    # Generate the cleaned address.
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated tokens.
    input_length = inputs["input_ids"].shape[-1]

    response_text = tokenizer.decode(
        generated[0][input_length:],
        skip_special_tokens=True
    )

    return parse_response(response_text)


print(f"Ready: {len(demonstrations) // 2} example pairs loaded.")

# TEST ONE ADDRESS BEFORE PROCESSING THE DATASET.
print("\nTesting the first address...")

test_result = extract_address(input_df.iloc[0]["raw_text"])

print("\nOriginal:")
print(input_df.iloc[0]["raw_text"])

print("\nQwen extraction:")
print(json.dumps(test_result, indent=2))
```

#### 6.1 Imports and shared variables

| Name | Role in this section |
| --- | --- |
| `json` | Converts example answers into JSON text and converts generated JSON text back into Python objects. |
| `re` | Removes complete thinking tags, searches for a possible JSON object, and checks the ZIP+4 pattern. |
| `time` | Imported here for the per-row timer used later in Section 7. |
| `torch` | Provides inference mode and the tensor-based model runtime. |
| `FIELDS` | Defines the five output field names and their order. |
| `SYSTEM_PROMPT` | Stores the instructions sent to the model. |
| `demonstrations` | Stores the example conversation messages created from the examples CSV. |

`FIELDS` is a tuple. Reusing it keeps the expected example answers, parsed results, and blank error results aligned. It does not include `id`, `raw_text`, `error`, or `seconds`, because those are added by Python rather than extracted by the model.

Changing `FIELDS` is a schema change, not a cosmetic rename. The examples upload validation, system prompt, and downstream expectations would also need to agree with the new fields.

#### 6.2 What the system prompt controls

`SYSTEM_PROMPT` is a multiline Python string. It tells the model what to extract, what to remove, how to choose between competing addresses, and when to leave a field empty.

The phrase “five string fields” requests a flat JSON object. Flat output is easier to convert into one spreadsheet row than nested objects or a paragraph.

The prompt's ambiguity rules matter: a row can successfully return an empty street address if the source contains only a city or two conflicting street addresses. An empty field can therefore be a valid extraction decision.

The code does not implement a separate deterministic rule engine for removing names, selecting a current address, or identifying a street. Those judgments are made by the model under these instructions and the supplied examples. Python performs only the parsing and limited cleanup described below.

#### 6.3 Building example conversations

`examples_df.iterrows()` yields each example row together with its DataFrame index. The loop uses `_` for the index because it is not needed, and `row` for the row's values.

For each example, the dictionary comprehension assigned to `expected` collects one value for every field in `FIELDS`. `str(row[field])` makes each value a string.

`demonstrations.extend(...)` adds **two separate message dictionaries** to the list:

| Message | Role | Content |
| --- | --- | --- |
| Example input | `user` | The literal prefix `Raw address: ` followed by the example's messy text. |
| Example answer | `assistant` | A JSON serialization of the expected five-field dictionary. |

The assistant message demonstrates the answer the model should imitate. It is supplied as part of the prompt, not generated during this loop.

`json.dumps(expected, ensure_ascii=False)` turns the dictionary into valid JSON text. It handles JSON quoting and escaping, while allowing non-ASCII characters to remain readable rather than converting them all to escape sequences.

If the CSV contains 8 examples, `demonstrations` contains 16 messages. This is why the status message divides its length by 2 to report the number of example pairs.

Every example is included for every input row. There is no similarity search or automatic selection of the most relevant examples. More examples also mean a longer prompt and repeated processing work.

#### 6.4 `parse_response(text)`: converting generated text into fields

This function accepts the model's decoded answer as text and returns a Python dictionary containing the five expected fields. Its processing order is significant.

**Step 1: Remove complete thinking blocks.**

`re.sub()` removes text matching `<think>.*?</think>`. The `.*?` match is non-greedy, and `re.DOTALL` lets the dot match newline characters. A complete tagged block can therefore span several lines. The following `.strip()` removes leading and trailing whitespace from the remaining answer.

This is a cleanup measure even though thinking is disabled in the chat-template call. It does not remove arbitrary commentary or an unfinished thinking block with no closing tag.

**Step 2: Try normal JSON parsing.**

`json.loads(text)` tries to convert the entire remaining response into a Python value. For the requested format, that value should be a dictionary.

Valid JSON has stricter syntax than informal Python-like text. Single-quoted keys, trailing commas, or a truncated closing brace can cause `json.JSONDecodeError`.

**Step 3: Try the fallback only if JSON decoding failed.**

The regular expression `r"\{[\s\S]*?\}"` looks for the first opening brace and the earliest following closing brace. `[\s\S]` matches whitespace and non-whitespace characters, allowing the match to span lines.

This can recover a simple object surrounded by extra text or Markdown formatting. If no brace-delimited candidate exists, the function raises `ValueError` with the first 200 characters of the cleaned response.

If a candidate exists, the function passes it to `json.loads()`. Invalid JSON in that candidate still raises an error. The fallback does not repair missing quotation marks, add missing braces, or retry generation.

The regex does not understand nested objects or braces inside quoted values. It may stop before the true end of an object in those cases. Flat JSON reduces this risk but does not eliminate it.

**Step 4: Require a dictionary.**

`isinstance(result, dict)` checks the parsed object's Python type. A JSON list, string, number, or `null` is rejected, even if it is syntactically valid JSON. The code raises `ValueError("Model response is not a JSON object.")`.

A syntactically valid JSON list does not trigger the fallback, because the original `json.loads()` call succeeded; it instead fails this dictionary check.

**Step 5: Build the cleaned dictionary.**

For each expected field, `result.get(field, "")` reads its value or supplies an empty string if the key is absent. `or ""` also changes falsy values, such as JSON `null`, `false`, or numeric zero, to an empty string. `str(...)` converts the remaining value to text, and `.strip()` removes surrounding whitespace.

Extra keys in the generated object are discarded. Missing expected keys become empty strings rather than triggering an error. A nonempty dictionary or list supplied as a field value is converted to its Python string representation, rather than rejected by a strict schema check.

Consequently, a valid but empty JSON object can pass parsing and produce five empty fields with no processing error. Successful parsing should not be treated as proof of meaningful extraction.

**Step 6: Uppercase state text.**

If `state` is nonempty, `.upper()` changes its capitalization. `mi` becomes `MI`, but `Michigan` becomes `MICHIGAN`; this step does not map full state names to abbreviations or check membership in a list of valid codes.

**Step 7: Shorten an exact ZIP+4 pattern.**

`re.fullmatch(r"\d{5}-\d{4}", cleaned["zip"])` checks the complete ZIP string. When it matches, `[:5]` keeps the first five characters. For example, `01234-5678` becomes `01234`.

The code does not pad a four-digit value, validate whether a ZIP exists, or convert a nine-digit value without a hyphen. The instruction to produce a five-digit ZIP is primarily enforced through prompting.

Finally, `return cleaned` gives the dictionary back to the caller.

#### 6.5 `extract_address(raw_text)`: one model request

This function handles one address string from start to finish. It builds a fresh conversation, tokenizes it, generates an answer, decodes the answer, and passes it to `parse_response()`.

**Build the message list.**

`messages` starts with one `system` message containing `SYSTEM_PROMPT`. `messages.extend(demonstrations)` then adds all example messages. Finally, `messages.append(...)` adds the current input as a new `user` message.

For 8 examples, one extraction request has 18 messages: 1 system message, 16 example messages, and 1 current-input message.

This fresh list prevents previous input records and their generated answers from accumulating in the conversation. The same model instance stays loaded, but prior records are not explicitly added to the next row's prompt.

**Apply the model's chat template and tokenize.**

A model needs numerical token IDs, not a Python list of message dictionaries. `tokenizer.apply_chat_template()` converts the conversation into the format expected by the tokenizer/model and tokenizes it.

| Argument | Why it is used |
| --- | --- |
| `messages` | Supplies the roles and text for the complete conversation. |
| `tokenize=True` | Requests token IDs rather than a formatted text string. |
| `add_generation_prompt=True` | Adds the template's marker for the assistant response to begin. |
| `return_tensors="pt"` | Requests PyTorch tensors. |
| `return_dict=True` | Requests a mapping-like collection of model inputs, including `input_ids`, rather than only a tensor. |
| `enable_thinking=False` | Requests the template's non-thinking mode where supported. |

`return_dict=True` is important because the generation call uses `**inputs` to pass named model arguments. The code also later accesses `inputs["input_ids"]` by name.

The number of tokens in this prompt includes the system instructions, every example input and answer, the current record, and the chat-template markers. `MAX_NEW_TOKENS` limits the answer length; it does not limit this input prompt.

**Move input tensors to the model device.**

`inputs.to(model.device)` moves the returned input tensors to the device reported by the model, normally a CUDA device in the intended runtime. This is the notebook's actual placement step; it does not provide custom input routing for every possible multi-device or offload arrangement.

**Generate the response.**

`torch.inference_mode()` disables autograd-related tracking for inference. It reduces unnecessary training-related overhead; it does not change the model's weights or fine-tune it.

`model.generate(**inputs, ...)` unpacks the input mapping into named arguments. `max_new_tokens=MAX_NEW_TOKENS` caps newly generated tokens at the configured limit, while generation can stop earlier. `do_sample=False` disables random sampling. `pad_token_id=tokenizer.eos_token_id` supplies the tokenizer's end-of-sequence ID as the padding ID for this call.

A response that reaches the limit may end before its JSON object is complete. That often appears later as a parsing error rather than as a dedicated truncation warning.

**Decode only the new tokens.**

For this causal language model generation path, the returned sequence contains the input token sequence followed by the generated tokens.

`inputs["input_ids"].shape[-1]` measures the prompt length in tokens. `generated[0]` selects the first sequence, and `[input_length:]` slices away its prompt portion. This matters because decoding the entire sequence would also include instructions and example JSON objects, which could confuse parsing.

`tokenizer.decode(..., skip_special_tokens=True)` converts the remaining token IDs into text while omitting special token markers recognized by the tokenizer. It is not itself a JSON validation step.

**Return parsed fields.**

The final line, `return parse_response(response_text)`, returns the cleaned dictionary. If generation or parsing raises an exception, this function does not catch it. The caller determines how that failure is handled.

#### 6.6 The first-row test

`input_df.iloc[0]` selects the first input row by position, and `["raw_text"]` selects its source text. The cell calls `extract_address()` once, prints the original, then prints the resulting dictionary as indented JSON.

This test checks that the complete prompt → tokenizer → model → decoder → parser path works for at least one record. It does not validate every row or measure dataset accuracy.

The test is not inside a `try/except` block. If it fails, the cell reports an exception directly. It also does not append to Section 7's results list or save a CSV. The first row is processed again when the main loop starts.

If you change the examples or prompt, rerun this section to rebuild `demonstrations` and retest. Merely uploading a different examples file does not rebuild a previously created demonstration list until this cell runs again.

### 7. Extract addresses

Section 6 defines what happens for one record. Section 7 applies that function to the selected input rows, adds identifiers and processing metadata, saves results, and prints a summary.

#### Complete Section 7 code

This is the complete notebook cell. It depends on the objects and functions initialized by the earlier sections.

```python
from pathlib import Path
from tqdm.auto import tqdm

MAX_ROWS = None
input_stem = Path(input_filename).stem
output_path = Path("/content") / f"{input_stem}_cleaned.csv"

subset = input_df if MAX_ROWS is None else input_df.head(MAX_ROWS)
results = []

for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Cleaning"):
    start = time.perf_counter()
    try:
        extracted = extract_address(row["raw_text"])
        error = ""
    except Exception as exc:
        extracted = {field: "" for field in FIELDS}
        error = f"{type(exc).__name__}: {str(exc)[:250]}"
    results.append({
        "id": row["id"],
        "raw_text": row["raw_text"],
        **extracted,
        "error": error,
        "seconds": round(time.perf_counter() - start, 2),
    })
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")

results_df = pd.DataFrame(results)
print(f"Saved {len(results_df)} rows to {output_path.name}")
print(f"Rows with processing errors: {results_df['error'].ne('').sum()}")
print(f"Average processing time: {results_df['seconds'].mean():.2f} sec/row")
display(results_df.head(10))
```

#### 7.1 Imports, row limit, and output filename

`Path` provides filesystem path handling. `tqdm.auto` wraps the row iterator with a progress display suited to the environment.

`MAX_ROWS = None` means that the main loop uses the entire input DataFrame. A positive integer selects the first rows in their existing order. This is not random sampling.

`Path(input_filename).stem` removes the final filename extension. If the upload is named `addresses.csv`, `input_stem` becomes `addresses`. The `/` operator on a `Path` joins `/content` with the filename `addresses_cleaned.csv`.

The filename is based on the input upload, not the examples upload. No date/time suffix is added. The cell assumes Colab's `/content` directory is available.

#### 7.2 Selecting rows and initializing results

The conditional expression assigned to `subset` means: use `input_df` if the row limit is `None`; otherwise use `input_df.head(MAX_ROWS)`.

The code then creates `results = []`, an empty Python list. Each iteration adds one dictionary representing one output row. This list stays in runtime memory and grows with the number of processed records.

Rerunning the cell executes `results = []` again. It does not load previously saved output or check which IDs have already been processed.

#### 7.3 Iterating with a progress bar

`subset.iterrows()` produces an iterator of `(index, row)` pairs. Wrapping it in `tqdm(..., total=len(subset), desc="Cleaning")` adds the progress bar and provides the expected total number of iterations.

The loop again uses `_` for the unused DataFrame index. It uses the actual `row["id"]` value for output identification, so a DataFrame's internal index is not substituted for the record ID.

The progress bar covers the main loop only. It does not include installation, model loading, or the first-row test. Its timing can include loop work such as CSV writes, whereas the stored `seconds` column measures a narrower interval.

#### 7.4 Starting the per-row timer

`start = time.perf_counter()` records a high-resolution elapsed-time reference immediately before extraction. It is suited to measuring durations and is not a calendar timestamp.

The later subtraction measures how long the extraction attempt took, including an unsuccessful attempt that raised an exception. The notebook does not record the date or time at which each row ran.

#### 7.5 Success and exception paths

The `try` block calls `extract_address(row["raw_text"])`. On success, `extracted` contains the five-field dictionary and `error` is set to an empty string.

If a standard Python exception derived from `Exception` is raised during that call, the `except` block creates an empty string for every field in `FIELDS`. It then formats an error message containing:

- `type(exc).__name__`: the exception class name, such as `ValueError` or `JSONDecodeError`.
- `str(exc)[:250]`: at most the first 250 characters of the exception's message.

The 250-character limit applies to the exception message portion, not the complete value including the class name and punctuation. The code does not store a full traceback or the full generated response for every failure.

| Outcome | Extracted fields | `error` | Next action |
| --- | --- | --- | --- |
| Extraction and parsing return | Returned five-field dictionary, possibly with some blank fields. | Empty string. | Append result and write CSV. |
| Extraction or parsing raises a caught exception | All five fields are blank. | Exception type and shortened message. | Append result and write CSV, then attempt the next row. |
| User interruption or runtime termination | Depends on when execution stopped. | No guaranteed new error row. | Previously written output may remain if the runtime survives. |
| CSV write fails | Result may already be in the in-memory list. | Write failure is not captured in this row's `error`. | The cell stops with an exception. |

The handler is scoped around `extract_address()`, not the whole iteration. It does not make a failed CUDA environment usable again, retry the row, or guarantee that later rows can succeed after a serious device error. `KeyboardInterrupt` and runtime termination are not ordinary extraction exceptions caught by this handler.

#### 7.6 Constructing one output row

`results.append({...})` adds one dictionary to the list. The dictionary combines values from three sources:

| Source | Fields |
| --- | --- |
| Original CSV row | `id`, `raw_text` |
| Model result after parsing, or blank error result | `address`, `city`, `state`, `zip`, `rationale` |
| Python processing metadata | `error`, `seconds` |

`**extracted` is dictionary unpacking. It inserts the extracted field names and their values directly into the output dictionary, so they become ordinary CSV columns rather than a nested `extracted` column.

`round(time.perf_counter() - start, 2)` stores elapsed time to two decimal places. This expression is evaluated before the subsequent CSV write. Time spent rewriting the output file is excluded from the recorded duration.

The row keeps its input ID even if extraction fails, allowing you to locate the record in the original dataset. The model is never asked to generate that ID.

#### 7.7 Writing all completed results after each row

`pd.DataFrame(results)` builds a temporary table from the entire accumulated list. `.to_csv(output_path, index=False, encoding="utf-8-sig")` writes that table to the same output path.

| Argument | Effect |
| --- | --- |
| `output_path` | Selects the file in `/content`. |
| `index=False` | Omits pandas' automatic row-index column from the export. The explicit `id` column is still included. |
| `encoding="utf-8-sig"` | Writes UTF-8 with a byte-order mark, which can help spreadsheet software recognize the text encoding. |

This is a **full rewrite**, not an append-only file operation. After row 1, the file contains 1 result; after row 2, it is rewritten with 2; after row 100, it is rewritten with 100.

Across `n` rows, the notebook serializes a cumulative total of `1 + 2 + ... + n` row records. That is `n(n+1)/2` row records written over the run, even though the final file contains only `n` rows. For 100 inputs, this is 5,050 row records serialized across all writes. This explains why saving frequently has increasing overhead on large datasets.

Frequent saving makes completed work accessible during a surviving session, but it is not automatic resume. It is also not an atomic checkpoint: interruption during a rewrite can leave an incomplete file. A storage error occurs outside the extraction handler and stops the cell.

#### 7.8 Building and displaying the final summary

After the loop, `results_df = pd.DataFrame(results)` creates the final in-memory table.

| Expression | What it reports |
| --- | --- |
| `len(results_df)` | Number of output rows accumulated, including rows with caught extraction errors. |
| `output_path.name` | Filename only, without the `/content` directory. |
| `results_df['error'].ne('').sum()` | Count of rows whose error value is not an empty string. `.ne('')` produces Boolean values, and summing counts the true values. |
| `results_df['seconds'].mean()` | Mean recorded extraction-attempt time, including caught failures. |
| `display(results_df.head(10))` | Notebook preview of up to the first ten results. |

The average is calculated from per-row values already rounded to two decimal places. It excludes model loading, the first-row test, and CSV write time. Fast failed rows can also lower the mean, so read it together with the processing-error count.

The preview does not limit the export to ten rows; it limits only what is displayed in the notebook.

The final summary assumes at least one result row exists. With no selected rows, the empty DataFrame lacks `error` and `seconds`, so the summary can fail. Use `None` or a positive integer for `MAX_ROWS` with the validated nonempty input file.

#### 7.9 One record from upload to export

For a given input row, the complete sequence is:

1. Section 7 reads its `raw_text` and starts the timer.
2. `extract_address()` combines the rules, all examples, and that text into a fresh conversation.
3. The tokenizer converts the conversation to model inputs.
4. The model generates an answer.
5. The code removes the prompt tokens and decodes the new answer tokens.
6. `parse_response()` parses and lightly normalizes the answer.
7. Section 7 combines the fields with the original ID/text, error status, and elapsed seconds.
8. The result is appended to the list, and the CSV is rewritten with every completed row.
9. The loop repeats for the next record.

The model produces the address fields and rationale. Python controls record identity, row order, parsing, timing, exception handling, and saving. Separating these responsibilities helps distinguish an incorrect extraction from a processing failure.

### 8. Download cleaned CSV

Checks that the output file exists and starts a browser download through Colab.

## Output file and columns

The output filename comes from the uploaded input filename:

**`<input_filename_without_extension>_cleaned.csv`**

For example, uploading `addresses_to_clean.csv` produces `/content/addresses_to_clean_cleaned.csv` inside Colab.

The CSV is written with UTF-8 encoding and a byte-order mark (`utf-8-sig`), without a pandas index column.

| Column | Meaning |
| --- | --- |
| `id` | Original record ID, or the generated sequential ID. |
| `raw_text` | Original input text. |
| `address` | Extracted primary street address. |
| `city` | Extracted city. |
| `state` | Extracted state, uppercased by the parser. |
| `zip` | Extracted ZIP; exact ZIP+4 patterns are shortened to five digits. |
| `rationale` | Model-generated explanation of changes, ambiguity, or missing information. |
| `error` | Empty when no processing exception was caught; otherwise the exception type and a shortened message. |
| `seconds` | Per-row elapsed extraction time, rounded to two decimal places. |

The `seconds` timer includes generation and parsing, but excludes the subsequent CSV write. The average shown at the end is therefore not the full wall-clock runtime per row.

When importing into Excel, set the ZIP column to **Text** to preserve leading zeros. CSV files do not carry spreadsheet cell-type formatting.

## Error handling and saved progress

Inside the main loop, extraction exceptions produce a row with blank extracted fields and an `error` message. The loop then continues to the next record.

A blank `error` means the extraction function returned successfully. It does **not** mean the address is correct, complete, deliverable, or independently verified.

The first-row test runs outside the loop's exception handler. If it fails, fix that failure before attempting the main extraction.

### What saving after each row provides

After each successful CSV write, the file contains all completed results from the current loop execution. If generation later stops while the runtime remains accessible, the existing partial file may still be downloadable.

### What it does not provide

- Automatic resume or skipping of previously processed IDs.
- Recovery after Colab discards the runtime and its local files.
- A separate checkpoint or backup file.
- Atomic writes or protection against interruption during a CSV write.

Rerunning the extraction cell resets `results` and starts again. It rewrites the same output path for the same input filename, including when switching from a subset test to a full run. Download any results you want to keep before rerunning.

## Performance and memory

Speed depends on the GPU, optional kernel compatibility, number and length of examples, length of each input, generated response length, and runtime resource availability. This README does not claim a benchmark or a guaranteed throughput.

For practical testing:

- Start with a small positive `MAX_ROWS` value.
- Keep examples relevant and concise; every example is repeated in every request.
- Review the first-row test before processing a larger file.
- Increase `MAX_NEW_TOKENS` if responses are being cut off, while allowing for more generation time and memory use.
- Reduce prompt length if you encounter context or memory errors.

There is no batching, automatic prompt truncation, example retrieval, explicit context-budget check, or built-in out-of-memory retry. Rewriting the growing CSV after every row also adds disk work as the dataset grows.

## Reviewing quality

Before relying on a full export, manually review a representative sample that includes both easy and difficult records.

Check whether:

- Street numbers and directionals were preserved.
- Recipient and unit details were removed as intended.
- The selected address matches an explicit current or physical address label.
- Missing information stayed blank instead of being guessed.
- Ambiguous records were handled consistently.
- ZIP codes kept their leading zeros.
- Records with no caught exception still contain plausible extractions.

For a more rigorous evaluation, prepare a separate manually labeled test set and compare output fields against those labels. Keep evaluation records separate from the examples supplied in the prompt. The notebook does not calculate accuracy, confidence scores, or agreement metrics.

The rationale is generated text. It can help with review, but it is not a confidence score or proof that an extraction is correct.

## Troubleshooting

| Symptom | Likely explanation | What to check |
| --- | --- | --- |
| `Select a GPU in Runtime → Change runtime type` | CUDA is unavailable to PyTorch. | Select a GPU runtime, reconnect, and rerun the cells in order. |
| `Select exactly one examples CSV` or `Select exactly one input CSV` | The upload dialog returned zero or multiple files. | Rerun that upload cell and choose one file. |
| Missing-column error | A required header is absent or spelled differently. | Check capitalization, spaces, and exact column names. |
| `Examples must contain nonempty raw_text values` | The examples file is empty or contains blank example inputs. | Remove blank rows or fill in their source text. |
| `Input CSV has no rows` | The CSV has headers but no records. | Upload a populated input file. |
| `Input id values must be unique` | IDs repeat. | Assign unique IDs, or remove the ID column to generate sequential ones. |
| CSV decoding or parsing error | Encoding or CSV structure does not match the reader. | Export as UTF-8 CSV and check quoting around commas or embedded line breaks. |
| `Qwen3_5ForCausalLM` import error | The active Transformers installation does not expose the required class, or imports are stale. | Check installation output and the active package version; restart the session if needed. |
| Optional wheel unavailable | No matching binary wheel was found. | Review the reported fallback and test actual model inference. |
| Optional module imports but inference raises a kernel error | Importability did not establish GPU compatibility. | Use a clean runtime without the incompatible optional extension, then retest. Disabling installation attempts does not uninstall an existing extension. |
| CUDA out-of-memory error | Model and runtime memory needs exceed available resources. | Start a fresh runtime, shorten examples or inputs, or use a runtime with more memory. |
| First-row test fails | Model loading, generation, or parsing failed before the loop. | Inspect the exception and the relevant preceding cell output. |
| JSON parsing errors | The response was malformed, unsuitable for the fallback parser, or truncated. | Review examples and instructions; consider a larger generation limit for truncated responses. |
| Output file missing | No CSV was successfully written at the expected path. | Run extraction and inspect its errors before downloading. |
| Output has fewer rows than expected | A subset was selected or processing stopped. | Check `MAX_ROWS`, the loop output, and whether the runtime was interrupted. |
| Results disappear after reconnecting | Colab replaced or discarded the runtime. | Download results before ending the session; automatic persistent storage is not configured. |

## Limitations

- Intended for US street-level extraction; the prompt and output schema are US-specific.
- Removes secondary unit details, so output may be unsuitable for mailing or identifying a specific apartment or suite.
- Does not geocode, check postal deliverability, or confirm city/state/ZIP consistency.
- Does not enforce valid two-letter state codes or reject every malformed ZIP value. Uppercasing and the exact ZIP+4 conversion are the only explicit state/ZIP cleanup steps.
- Does not enforce every prompt rule in Python; the model can miss details or invent information despite instructions.
- Does not use constrained JSON generation or a strict output schema validator.
- Does not preserve extra input columns in the exported CSV.
- Does not automatically retry failures, resume interrupted work, or save to Google Drive.
- Runs sequentially and can be slow on large datasets.
- Requires adaptation for local Jupyter, Windows Python, or other non-Colab environments because upload/download and output paths are Colab-specific.

## Data handling and publishing

Use synthetic or appropriately de-identified records for public demonstrations. Uploaded address text and examples are processed in Colab, and the cleaned CSV retains the original `raw_text` alongside the extracted fields.

Before publishing the notebook on GitHub, review and clear saved cell outputs that contain uploaded records, previews, or extracted addresses. Notebook files can retain those outputs even when the CSV files are not included in the repository.

The repository can contain the notebook and this README without containing your private input files, model cache, or generated results. If you add examples, label synthetic examples clearly.

This workflow has no paid model API calls in its extraction code. GPU availability and any Colab usage costs depend on your own access and plan; the notebook does not guarantee a particular GPU or uninterrupted session.

## Reproducibility

Record the GPU, Python environment, PyTorch/CUDA versions, installed package versions, model revision, settings, examples CSV, and input CSV used for a run.

The notebook pins pandas, uses version ranges for several other packages, leaves some dependencies unpinned, and does not pin a model revision. A fresh run at a later date can therefore load a different software environment or model revision.

Disabling sampling reduces one source of variation, but does not guarantee identical results across GPUs, library versions, kernels, or model revisions. No formal accuracy benchmark or cross-environment reproducibility test is included.


