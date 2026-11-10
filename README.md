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

Builds the prompt examples, defines the response parser and extraction function, and tests the first input row.

The parser:

1. Removes complete `<think>...</think>` blocks if present.
2. Attempts to parse the full response as JSON.
3. If that fails, attempts to parse the first brace-delimited object-like substring.
4. Requires the parsed result to be a dictionary.
5. Keeps the five expected fields, converts values to strings, and strips surrounding whitespace. Missing or falsy values become empty strings.
6. Uppercases the state value.
7. Converts a ZIP matching `12345-6789` to its first five digits.

The fallback substring search is a simple regular expression, not a general parser for nested or malformed JSON. The parser does not enforce an address schema beyond these steps.

### 7. Extract addresses

Processes the selected rows sequentially. For every row it measures elapsed time, attempts extraction, records an exception if one occurs, appends the result, and rewrites the CSV with all results completed so far.

At the end, the cell displays the saved row count, number of processing errors, average recorded seconds per row, and a preview of the first ten results.

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
