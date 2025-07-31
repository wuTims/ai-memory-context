# AI Memory Context Tool

This guide explains how to use the AI Memory Context tool, which helps provide relevant contextual information to AI assistants based on your specific needs.

## Overview

The AI Memory Context tool allows you to:

1. Store knowledge as tagged memory records
2. Retrieve relevant memories based on natural language queries
3. Filter memories by specific tags (e.g., "cardano", "style")
4. Provide focused context to AI assistants for more accurate responses

## Directory Structure

```
ai-memory-context/
├── .venv/                 # Python virtual environment
├── memory/                # JSON memory records
│   ├── YYYY-MM-DD_id.json # Individual memory records
│   └── *_mega-summary.json # Summarized older memories
├── indices/               # Vector search indices
│   ├── memory.index       # FAISS search index
│   ├── docs.npy           # Memory record metadata
│   └── vectors.npy        # Embedded vectors
├── scripts/               # Python scripts
│   ├── assemble.py        # Main script for context assembly
│   ├── build_index.py     # Creates search indices
│   ├── retrieve.py        # Retrieves relevant memories
│   └── summarize.py       # Summarizes older memories
└── setup.md               # This guide
```

## Prerequisites

1. Python 3.8+ with virtual environment
2. Required packages: `sentence-transformers`, `faiss-cpu`, `numpy`
3. Memory records in JSON format

## Setup Instructions

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install sentence-transformers faiss-cpu numpy
   ```

3. **Build the index** (after adding memory records):
   ```bash
   cd ~/ai-memory-context
   source .venv/bin/activate
   python scripts/build_index.py
   ```

## Using the Context Tool

### Basic Usage

The main script for using the context tool is `assemble.py`, which takes a user prompt and returns relevant context:

```bash
cd ~/ai-memory-context
source .venv/bin/activate
python scripts/assemble.py "your question or prompt here"
```

### Filtering by Tags

To filter memories by specific tags (e.g., only "cardano" or "style" related memories):

```bash
python scripts/assemble.py --filter-tags cardano "your Cardano-specific question"
python scripts/assemble.py --filter-tags style "your style-related question"
```

You can also use this in workflows by creating shell scripts or aliases.

### Workflow Integration

The tool is designed to be used in workflows, such as:

1. **Cardano Context Workflow**:
   ```bash
   cd ~/ai-memory-context
   source .venv/bin/activate
   python scripts/assemble.py --filter-tags cardano "your prompt"
   ```

2. **Style Context Workflow**:
   ```bash
   cd ~/ai-memory-context
   source .venv/bin/activate
   python scripts/assemble.py --filter-tags style "your prompt"
   ```

## Memory Record Format

Memory records are stored as JSON files in the `memory/` directory with the following structure:

```json
{
  "id": "unique-identifier",
  "type": "concept|pattern|rule|warning|setup",
  "name": "Short descriptive name",
  "description": "Detailed explanation of the memory",
  "rationale": "Why this is important",
  "pattern": "Example code or pattern",
  "tags": ["tag1", "tag2", "cardano", "style"],
  "date": "2025-07-30"
}
```

## How It Works

1. **Embedding and Indexing**:
   - Memory records are embedded using sentence-transformers
   - FAISS is used for efficient vector similarity search
   - Records are indexed by their semantic content

2. **Context Assembly**:
   - When you run `assemble.py` with a prompt:
     - The prompt is embedded into a vector
     - Similar memory records are retrieved
     - Records are filtered by tags if specified
     - Context is assembled with mega-summaries, recent records, and relevant records
     - The result is formatted for the AI assistant

3. **Output Structure**:
   The assembled context includes:
   - Mega-summaries of older memories
   - Core principles (style or Cardano)
   - Recent memory records
   - Relevant memory records based on your query
   - Your original prompt

## Advanced Usage

### Creating New Memory Records

1. Create a JSON file in the `memory/` directory with the required fields
2. Run `build_index.py` to update the search indices

### Summarizing Older Memories

Run the summarization script to create condensed versions of older memories:

```bash
python scripts/summarize.py
```

### Custom Filtering

You can combine multiple tag filters:

```bash
python scripts/assemble.py --filter-tags cardano --filter-tags defi "your prompt"
```

## Troubleshooting

- **Missing dependencies**: Ensure all required packages are installed
- **Index errors**: Rebuild the index if you encounter search issues
- **Empty results**: Check that your memory records have appropriate tags

## Example

```bash
# Get Cardano-specific context for wallet development
python scripts/assemble.py --filter-tags cardano "How do I build a Cardano wallet?"

# Get style guidance for React components
python scripts/assemble.py --filter-tags style "How should I structure React components?"
```

The output will provide relevant context that can be fed to an AI assistant for more informed responses.
