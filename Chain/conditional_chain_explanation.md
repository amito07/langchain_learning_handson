# Conditional Chain Explanation

## Overview
This script demonstrates how to create a **conditional routing chain** in LangChain that analyzes text sentiment and routes it to different response generators based on whether the sentiment is positive, negative, or neutral.

---

## Step-by-Step Breakdown

### Step 1: Import Required Libraries
```python
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from typing import Literal
```

**What's happening:**
- `StrOutputParser`: Parses LLM output as plain strings
- `PydanticOutputParser`: Parses LLM output into structured Pydantic models
- `ChatGoogleGenerativeAI`: Google's Gemini model interface
- `RunnableBranch`: Enables conditional routing based on input
- `RunnablePassthrough`: Passes input data through unchanged
- `RunnableLambda`: Wraps Python functions to use in chains
- `Literal`: TypeScript-like literal types for strict type checking

---

### Step 2: Load Environment Variables and Initialize Model
```python
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
```

**What's happening:**
- Loads API keys from `.env` file (contains Google API credentials)
- Initializes the Gemini 2.5 Flash model for both classification and response generation

---

### Step 3: Define Sentiment Classification Schema
```python
class Condition(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the text"
    )
```

**What's happening:**
- Creates a Pydantic model that enforces strict typing
- `Literal["positive", "negative", "neutral"]` means the sentiment can ONLY be one of these three values
- This structure will be used to parse the LLM's sentiment analysis output

---

### Step 4: Initialize Parsers
```python
parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Condition)
```

**What's happening:**
- `parser`: Used for converting LLM responses to plain text strings
- `pydantic_parser`: Converts LLM responses into structured `Condition` objects

---

### Step 5: Create Sentiment Classification Prompt
```python
template1 = PromptTemplate(
    template="Analyze the sentiment of the following text and classify it as positive, negative, or neutral: \n {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)
```

**What's happening:**
- Creates a prompt template that asks the LLM to analyze sentiment
- `{text}`: Placeholder for the input text to analyze
- `{format_instructions}`: Automatically generated instructions telling the LLM how to format its output as a JSON matching the `Condition` schema

**Example of what gets sent to the LLM:**
```
Analyze the sentiment of the following text and classify it as positive, negative, or neutral:
I love this product!
The output should be formatted as a JSON instance that conforms to the JSON schema below...
```

---

### Step 6: Build Classification Chain
```python
classification_chain = template1 | model | pydantic_parser
```

**What's happening:**
- **Chain flow:** `Input {"text": "..."} → template1 → model → pydantic_parser → Condition object`
- The `|` operator pipes data through each component
- Output: A `Condition` object with the sentiment field populated

**Example:**
```python
Input: {"text": "I love this!"}
Output: Condition(sentiment="positive")
```

---

### Step 7: Create Helper Function to Preserve Data
```python
def add_sentiment(inputs):
    original_text = inputs["text"]["text"]  # Extract the text from nested dict
    sentiment_obj = inputs["sentiment"]
    return {"text": original_text, "sentiment": sentiment_obj.sentiment}
```

**What's happening:**
- This function receives a dictionary with both the original text and the sentiment analysis result
- It flattens the nested structure and extracts the sentiment value from the Pydantic object
- **Input:** `{"text": {"text": "I love this!"}, "sentiment": Condition(sentiment="positive")}`
- **Output:** `{"text": "I love this!", "sentiment": "positive"}`

**Why is this needed?**
The text is nested because `RunnablePassthrough()` passes the entire input dict, creating `{"text": {"text": "..."}}`. This function flattens it for easier access downstream.

---

### Step 8: Define Response Templates for Each Sentiment
```python
positive_propmt = PromptTemplate(
    template="Write an appropriate response to the following text: \n {text}",
    input_variables=["text"]
)

negative_propmt = PromptTemplate(
    template="Write an appropriate negative response to the following text: \n {text}",
    input_variables=["text"]
)

neutral_propmt = PromptTemplate(
    template="Write an appropriate neutral response to the following text: \n {text}",
    input_variables=["text"]
)
```

**What's happening:**
- Creates three different prompt templates, one for each sentiment type
- Each expects the original `{text}` as input
- The LLM will generate contextually appropriate responses based on the sentiment

---

### Step 9: Create Response Generation Chains
```python
positive_chain = positive_propmt | model | parser
negative_chain = negative_propmt | model | parser
neutral_chain = neutral_propmt | model | parser
```

**What's happening:**
- Each chain: `Input {"text": "..."} → template → model → parser → string response`
- All three chains have the same structure but use different prompts

---

### Step 10: Create Conditional Branch
```python
branch = RunnableBranch(
    (lambda x: x["sentiment"] == 'positive', positive_chain),
    (lambda x: x["sentiment"] == 'negative', negative_chain),
    neutral_chain  # default case
)
```

**What's happening:**
- `RunnableBranch` evaluates conditions in order and routes to the first matching chain
- **Condition 1:** If `sentiment == "positive"` → use `positive_chain`
- **Condition 2:** If `sentiment == "negative"` → use `negative_chain`
- **Default:** Otherwise (neutral) → use `neutral_chain`
- The last argument (without a lambda) is the **default case**

**Data flow through branch:**
```python
Input: {"text": "I love this!", "sentiment": "positive"}
→ Condition 1 matches → positive_chain executes
→ Output: "Thank you for your wonderful feedback! We're thrilled..."
```

---

### Step 11: Build Complete Conditional Chain
```python
conditional_chain = (
    {"text": RunnablePassthrough(), "sentiment": classification_chain}
    | RunnableLambda(add_sentiment)
    | branch
)
```

**What's happening - Breaking it down:**

**Part 1:** Parallel execution
```python
{"text": RunnablePassthrough(), "sentiment": classification_chain}
```
- Takes input `{"text": "I love this!"}`
- **"text" branch:** `RunnablePassthrough()` passes input unchanged → `{"text": "I love this!"}`
- **"sentiment" branch:** Runs `classification_chain` → `Condition(sentiment="positive")`
- **Combined output:** `{"text": {"text": "I love this!"}, "sentiment": Condition(sentiment="positive")}`

**Part 2:** Data transformation
```python
| RunnableLambda(add_sentiment)
```
- Flattens the nested structure
- Output: `{"text": "I love this!", "sentiment": "positive"}`

**Part 3:** Conditional routing
```python
| branch
```
- Evaluates sentiment and routes to appropriate response chain
- Output: Final string response from the selected chain

---

## Complete Data Flow Diagram

```
Input: {"text": "I love this product!"}
         ↓
    ┌────┴────┐
    │ Parallel │
    │Execution │
    └────┬────┘
         │
    ┌────┴─────────────────────────┐
    │                              │
RunnablePassthrough()    classification_chain
    │                              │
{"text": {"text": "..."}}   Condition(sentiment="positive")
    │                              │
    └────────┬─────────────────────┘
             │
    {"text": {...}, "sentiment": Condition(...)}
             │
             ↓
      add_sentiment()
             │
    {"text": "I love...", "sentiment": "positive"}
             │
             ↓
         branch
             │
      (checks sentiment)
             │
      sentiment == "positive" ✓
             │
             ↓
      positive_chain
             │
             ↓
   "Thank you for your wonderful feedback!..."
```

---

### Step 12: Execute the Chain
```python
text = '''I love this product! It has exceeded all my expectations...'''
result = conditional_chain.invoke({"text": text})
print("Result ===> ", result)
```

**What's happening:**
- Invokes the entire chain with the input text
- The chain automatically:
  1. Analyzes sentiment (positive)
  2. Routes to the positive response chain
  3. Generates an appropriate positive response
- Prints the final result

---

## Key Concepts

### 1. **RunnablePassthrough()**
Preserves the original input data so it's available later in the chain.

### 2. **Parallel Execution**
```python
{"text": RunnablePassthrough(), "sentiment": classification_chain}
```
Both operations run and their results are combined into a single dictionary.

### 3. **Conditional Routing**
`RunnableBranch` evaluates conditions sequentially and executes the first matching branch.

### 4. **Data Preservation**
The `add_sentiment()` function ensures both the original text and the classification result are available for downstream processing.

---

## Why This Architecture?

1. **Efficient:** Sentiment analysis happens once, then routes to specialized handlers
2. **Maintainable:** Each sentiment type has its own clear response logic
3. **Extensible:** Easy to add new sentiment types or modify existing responses
4. **Type-Safe:** Pydantic ensures sentiment is always one of the three valid values

---

## Potential Improvements

1. **Add logging** to track which branch is taken
2. **Add error handling** for malformed LLM outputs
3. **Cache sentiment analysis** for repeated texts
4. **Add more granular sentiments** (very positive, slightly negative, etc.)
5. **Use streaming** for real-time responses on long texts
