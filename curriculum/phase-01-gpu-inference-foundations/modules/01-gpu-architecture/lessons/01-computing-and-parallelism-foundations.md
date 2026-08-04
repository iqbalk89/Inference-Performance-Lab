# Lesson 01 — Computing and Parallelism Foundations

**Estimated study time:** 4–5 hours across multiple sittings

**Prerequisites:** None

## Chapter Purpose

A GPU is not simply a faster CPU. A CPU and a GPU are processors designed with
different priorities. Understanding those priorities requires a foundation in
how programs become executable work, how processors obtain data, which
operations must occur in order, which operations may occur concurrently, and
how latency differs from throughput.

This chapter assumes no programming, computer-architecture, linear-algebra, or
machine-learning background. It deliberately explains ideas that experienced
engineers may take for granted.

## Learning Objectives

By the end of this chapter, you should be able to:

1. Distinguish source code, a program, an instruction, and data.
2. Describe a simplified fetch → decode → execute cycle.
3. Trace a list-addition operation through loads, arithmetic, and stores.
4. Distinguish dependency order from the order selected by an implementation.
5. Distinguish latency, throughput, capacity, concurrency, and parallelism.
6. Explain the broad design priorities of CPUs and GPUs.
7. Work through a small matrix multiplication and identify both its dependent
   and independent operations.
8. Explain why transformer inference creates useful GPU work.
9. Explain at a high level why prefill and decode expose different amounts and
   shapes of parallel work.
10. Identify situations in which using a GPU can make a complete workload
    slower rather than faster.

---

## 1. From a Human Idea to Processor Work

### 1.1 A computer transforms information

At a very high level, computing means transforming input information into
output information according to a defined procedure.

```mermaid
flowchart LR
    IN[Input information] --> PROC[Defined procedure]
    PROC --> OUT[Output information]
```

Examples:

| Input | Procedure | Output |
| --- | --- | --- |
| Two numbers | Add them | Their sum |
| Image pixels | Resize algorithm | Smaller image |
| Prompt tokens | Transformer inference | Next-token scores |
| Network request | Request handler | HTTP response |

The **procedure** is what a program describes. The processor performs the
low-level operations needed to carry it out.

### 1.2 Source code is written for humans and software tools

Suppose a programmer writes:

```python
total = left + right
```

This is **source code**. It communicates an intention using the rules of the
Python language. A physical processor does not directly understand the words
`total`, `left`, or `right`. Other software must translate or interpret that
intent and eventually arrange operations the processor understands.

The complete route varies by language and runtime. A simplified picture is:

```mermaid
flowchart TD
    IDEA[Human intention] --> SRC[Source code]
    SRC --> LANG[Compiler, interpreter, or runtime]
    LANG --> INST[Processor instructions]
    INST --> HW[Processor execution]
    HW --> RESULT[Result]
```

Python often involves an interpreter and compiled native libraries. PyTorch
adds further dispatch and library layers. Lesson 06 explains that software
stack in detail. For this chapter, the important distinction is:

> A source-code statement expresses what should happen. Processor instructions
> are the much smaller operations used to make it happen.

### 1.3 What is an instruction?

An **instruction** is an encoded operation a particular processor architecture
defines. Typical instruction categories include:

- **Load:** copy data from memory into storage close to the processor.
- **Store:** copy a result from processor-local storage into memory.
- **Arithmetic:** add, subtract, multiply, or divide values.
- **Logical:** combine or test bit patterns.
- **Compare:** determine relationships such as equal, less than, or greater
  than.
- **Branch or jump:** select a different next instruction.

One source statement can require several instructions. The exact instructions
depend on language implementation, processor architecture, optimization, data
type, and surrounding code.

Conceptually, `total = left + right` might require:

```text
1. LOAD  the value of left  into a processor register.
2. LOAD  the value of right into another register.
3. ADD   the two register values.
4. STORE the sum in the memory location associated with total.
```

This is a teaching model, not literal Python machine code. Python objects and
runtime behavior add more steps. The model isolates the essential movement and
arithmetic.

### 1.4 What is data?

**Data** is the encoded information instructions read or modify. It includes:

- Numbers
- Characters and text
- Memory addresses
- Images and audio samples
- Tensor values
- Model weights
- Instructions themselves

Everything is represented as bits in hardware. A data type tells software how
to interpret a bit pattern—for example, as an integer or floating-point value.

The same operation name can behave differently for different data types. Adding
two integers, two floating-point values, and two Python strings are different
operations even though source code may use `+` for each.

### 1.5 What is a program?

A **program** is more than a loose list of instructions. It is an organized
description of:

- Which operations should be performed
- Which data they use
- Which results become inputs to later operations
- Which decisions alter the path
- When repetition stops

The order of a program is constrained by **dependencies**. If operation B needs
the result of operation A, B cannot correctly finish before A produces that
result.

```mermaid
flowchart LR
    A[Read two inputs] --> B[Add inputs]
    B --> C[Store sum]
```

The arrows mean “must happen before because the later step needs the earlier
result.”

### 1.6 A simplified instruction cycle

A traditional introductory model says a processor repeatedly:

1. **Fetches** the next instruction.
2. **Decodes** what the instruction requests.
3. **Obtains operands**, such as register values or data loaded from memory.
4. **Executes** the requested operation.
5. **Writes back** or stores the result.
6. Selects the next instruction and repeats.

```mermaid
flowchart LR
    F[Fetch instruction] --> D[Decode instruction]
    D --> O[Obtain operands]
    O --> E[Execute]
    E --> W[Write result]
    W --> N[Select next instruction]
    N --> F
```

Real CPUs overlap, reorder, and pipeline parts of this process. Real GPUs issue
instructions across groups of threads. The cycle remains a useful conceptual
starting point because every arithmetic result still requires an operation and
available input data.

### 1.7 Complete worked example: adding two lists

Consider elementwise list addition:

```text
Input A = [2, 4, 6, 8]
Input B = [1, 3, 5, 7]
Output C should become [3, 7, 11, 15]
```

The intended rule is:

```text
C[i] = A[i] + B[i]
```

Here, `i` is an **index** identifying one position. The output list `C` is not
known in advance; the computer must calculate and store it.

#### One possible sequential implementation

```text
for i from 0 through 3:
    C[i] = A[i] + B[i]
```

For `i = 0`, a simplified execution story is:

```text
1. Determine where A[0] is stored.
2. Load A[0], which is 2.
3. Determine where B[0] is stored.
4. Load B[0], which is 1.
5. Add 2 + 1 to produce 3.
6. Determine where C[0] should be stored.
7. Store 3 in C[0].
8. Advance i from 0 to 1.
9. Check whether another iteration is required.
```

The processor then performs the equivalent work for `i = 1`, `i = 2`, and
`i = 3`.

```text
Iteration 0: load 2, load 1, add →  3, store C[0]
Iteration 1: load 4, load 3, add →  7, store C[1]
Iteration 2: load 6, load 5, add → 11, store C[2]
Iteration 3: load 8, load 7, add → 15, store C[3]
```

Within one iteration, some order is necessary:

```mermaid
flowchart LR
    LA[Load A i] --> ADD[Add values]
    LB[Load B i] --> ADD
    ADD --> SC[Store C i]
```

The addition needs both input values. The store needs the sum. Therefore the
store cannot correctly occur before the addition.

#### Why the four additions are independent

Now compare the four iterations:

```mermaid
flowchart LR
    A0[Load A0 and B0] --> X0[Add] --> C0[Store C0]
    A1[Load A1 and B1] --> X1[Add] --> C1[Store C1]
    A2[Load A2 and B2] --> X2[Add] --> C2[Store C2]
    A3[Load A3 and B3] --> X3[Add] --> C3[Store C3]
```

There are dependencies **within** each row, but no result arrow between rows:

- Computing `C[0]` requires `A[0]` and `B[0]`.
- Computing `C[1]` requires `A[1]` and `B[1]`.
- `C[1]` does not require the value of `C[0]`.
- `C[0]` does not require the value of `C[1]`.

That is what independence means here. It does **not** mean the operations must
run simultaneously. It means they may be scheduled in any order, or
concurrently, without changing the mathematically correct result—provided each
output position is written correctly.

Possible valid execution orders include:

```text
Sequential:       C[0], C[1], C[2], C[3]
Reverse:          C[3], C[2], C[1], C[0]
Two at a time:    C[0] and C[1], then C[2] and C[3]
All concurrently: C[0], C[1], C[2], and C[3]
```

The implementation and available hardware determine which schedule is used.

#### Mathematical independence versus physical execution

This distinction is foundational:

- **Mathematical independence** describes which results depend on which other
  results.
- **Execution scheduling** describes when hardware actually performs ready
  work.
- **Parallel hardware** describes how much work can physically execute at once.

Four independent additions on one simple arithmetic unit may still execute one
after another. Four independent additions on suitable parallel hardware may
execute together. One million independent additions may execute in many waves
because hardware resources are finite.

#### What if one output depended on the previous output?

Consider a running sum:

```text
C[0] = A[0]
C[1] = C[0] + A[1]
C[2] = C[1] + A[2]
C[3] = C[2] + A[3]
```

Now the dependency graph is a chain:

```mermaid
flowchart LR
    C0[C0] --> C1[C1]
    C1 --> C2[C2]
    C2 --> C3[C3]
```

The straightforward algorithm cannot calculate `C[3]` before `C[2]`, because
`C[3]` needs `C[2]`. This is fundamentally different from elementwise list
addition.

### Section 1 checkpoint

You should now be able to explain:

- Why source code is not identical to processor instructions
- Why an addition often requires loads and a later store
- Which ordering constraints exist inside one elementwise addition
- Why different output elements are mathematically independent
- Why independence permits parallel scheduling but does not guarantee it

---

## 2. Processor, Registers, and Memory

### 2.1 Separate responsibilities

A **processor** executes instructions. **Memory** retains encoded instructions
and data. Neither job is useful alone: a processor needs data and instructions,
while stored data does not transform itself.

```mermaid
flowchart LR
    MEM[Memory<br/>instructions and data]
    CPU[Processor<br/>executes operations]
    MEM -- instruction and data loads --> CPU
    CPU -- result stores --> MEM
```

### 2.2 Memory locations and addresses

You can think of main memory as a very large collection of numbered storage
locations. A number identifying a location is an **address**.

```text
Address      Stored value
1000         2
1004         4
1008         6
1012         8
```

The addresses are illustrative. A real layout depends on data type, runtime,
alignment, and representation.

For a compact fixed-width array, software can often find an element using a
base address and an index:

```text
element address = base address + index × bytes per element
```

If `A` begins at address 1000 and each value occupies 4 bytes:

```text
A[0] address = 1000 + 0 × 4 = 1000
A[1] address = 1000 + 1 × 4 = 1004
A[2] address = 1000 + 2 × 4 = 1008
```

This address calculation is one of the steps hidden by a high-level expression
such as `A[i]`.

> Python lists are collections of references to Python objects and are more
> complex than this compact-array picture. Tensors and low-level arrays more
> closely resemble contiguous typed storage. The simplified model teaches the
> memory relationship without pretending to describe Python object internals.

### 2.3 Registers: the processor's immediate workspace

Processors contain very small, very fast storage locations called
**registers**. Arithmetic instructions generally operate on values available
in registers or other processor-local pathways, not on an abstract variable
name in source code.

```mermaid
flowchart LR
    MEM[Memory] -- load --> R1[Register 1]
    MEM -- load --> R2[Register 2]
    R1 --> ALU[Arithmetic unit]
    R2 --> ALU
    ALU --> R3[Result register]
    R3 -- store --> MEM
```

Registers are fast but scarce. Main memory is much larger but farther from
execution.

### 2.4 Why data movement matters

Suppose an arithmetic unit can perform an addition in one unit of time, but
obtaining the values takes many units of time. The arithmetic unit cannot add
values it has not received.

```text
Time ───────────────────────────────────────────▶
Load A:       [---------- waiting ----------]
Load B:                    [----- waiting -----]
Add:                                          [x]
Store:                                          [----]
```

The one-step arithmetic is not the dominant cost. Data movement and waiting
are.

### 2.5 Caches: keeping likely-needed data closer

Between registers and large memory, processors use **caches**: smaller, faster
storage that retains recently or nearby accessed data. If required data is
already in a cache, the processor may avoid a longer trip to main memory.

```text
Closest, smallest, usually lowest latency
        Registers
            ↓
           Cache
            ↓
        Main memory
Farthest, largest, usually higher latency
```

This hierarchy exists in different forms on CPUs and GPUs. Lesson 04 examines
GPU registers, shared memory/L1, L2, and VRAM in detail.

### 2.6 Waiting does not always mean total idleness

Modern processors try to perform other useful work while one operation waits.
A CPU may execute independent instructions out of order. A GPU may issue an
instruction from another ready group of threads. These mechanisms do not make
memory latency disappear. They attempt to **hide** it behind other work.

```mermaid
sequenceDiagram
    participant WorkA as Work A
    participant Processor
    participant WorkB as Work B
    WorkA->>Processor: Request data
    Note over WorkA: Waiting
    Processor->>WorkB: Execute independent ready work
    WorkA-->>Processor: Data arrives
    Processor->>WorkA: Resume
```

### 2.7 Capacity, bandwidth, and latency preview

These are separate properties:

- **Capacity:** how much data can be stored.
- **Bandwidth:** how much data can be transferred per unit time.
- **Latency:** how long one access or operation takes.

A memory system can have large capacity and high bandwidth while an individual
access still has meaningful latency. Lesson 04 develops these distinctions with
calculations.

### Section 2 checkpoint

Explain why the statement “the processor adds the numbers” omits the critical
questions of where the numbers are stored, how they reach the arithmetic unit,
and where the result goes.

---

## 3. Latency, Throughput, Capacity, Concurrency, and Parallelism

Performance discussions become confusing when these words are treated as
synonyms. They answer different questions.

### 3.1 Latency

**Latency** is the elapsed time for one defined unit of work.

Examples:

- Time for one memory request
- Time for one matrix multiplication
- Time from an inference request arriving to its first generated token
- Time for a complete response

A latency statement must define start, end, and units:

```text
Poor:  “Latency is 40.”
Better: “Warm end-to-end request latency is 40 milliseconds.”
```

### 3.2 Throughput

**Throughput** is completed work divided by elapsed time.

```text
throughput = completed units / elapsed time
```

Examples:

- Requests per second
- Output tokens per second
- Bytes transferred per second
- Arithmetic operations per second

If a server completes 240 requests in 60 seconds:

```text
throughput = 240 requests / 60 seconds = 4 requests per second
```

This does not reveal the latency experienced by each request. Some may have
waited much longer than others.

### 3.3 Capacity

**Capacity** is how much can be held or accommodated, not how quickly work is
completed.

Examples:

- 24 GB of VRAM
- A queue that holds 1,000 requests
- A batch containing 32 sequences

Capacity can enable throughput, but it is not throughput.

### 3.4 Concurrency and parallelism

**Concurrency** means multiple tasks are in progress during overlapping periods
of time. They do not necessarily execute at the same instant.

**Parallelism** means multiple operations actually execute at the same time on
different resources.

One worker alternating between tasks creates concurrency without physical
parallel execution:

```text
Time ───────────────────────────────────────────▶
Worker 1: [Task A][Task B][Task A][Task B]
```

Two workers can execute in parallel:

```text
Time ───────────────────────────────────────────▶
Worker 1: [──────────── Task A ────────────]
Worker 2: [──────────── Task B ────────────]
```

Software discussions sometimes use these words loosely, but the distinction is
useful when reasoning about hardware.

### 3.5 Worked service example

Imagine one worker takes 100 ms to process a request.

With no overlap:

```text
Request A: [100 ms]
Request B:          [100 ms]
Request C:                   [100 ms]
```

Idealized throughput is 10 requests per second, and each request's service time
is 100 ms, excluding queueing.

Now imagine four workers process four requests simultaneously:

```text
Worker 1: [Request A: 100 ms]
Worker 2: [Request B: 100 ms]
Worker 3: [Request C: 100 ms]
Worker 4: [Request D: 100 ms]
```

Four requests finish after approximately 100 ms. Aggregate throughput rises to
about 40 requests per second while the service latency of each request remains
about 100 ms.

But if requests wait 80 ms for a batch to form, client-observed latency becomes
approximately:

```text
80 ms queueing + 100 ms processing = 180 ms
```

Throughput improved while individual end-to-end latency worsened.

### 3.6 The ferry analogy, completed carefully

Imagine:

```text
Vehicle      Trip latency      Passengers per trip      Trips per hour
Speedboat       10 min                  4                     6
Ferry           20 min                100                     3
```

Idealized passenger throughput:

```text
Speedboat: 4 × 6   = 24 passengers/hour
Ferry:     100 × 3 = 300 passengers/hour
```

The speedboat gives one passenger a shorter crossing latency. The ferry moves
far more passengers per hour. This resembles the broad CPU/GPU tradeoff, but
only as an analogy: processors do not literally wait to fill boats, and their
workloads have dependencies and memory behavior.

### 3.7 Why this matters for inference

An inference system may optimize:

- Time to first token for one user
- Time per output token
- Requests completed per second
- Tokens generated per second across all users
- Maximum concurrent sequences

These goals can conflict. An engineer must state which metric matters.

### Section 3 checkpoint

Construct an example in which throughput increases but end-to-end latency also
increases. Identify queueing time and processing time separately.

---

## 4. Dependencies, Order, and Parallel Work

### 4.1 Dependency order

An operation is **dependent** on another when it needs the earlier operation's
result or when both operations access shared state in an order that affects
correctness.

Example:

```text
x = 2
y = x + 3
z = y × 4
answer = z - 1
```

```mermaid
flowchart LR
    X[x = 2] --> Y[y = x + 3]
    Y --> Z[z = y × 4]
    Z --> A[answer = z - 1]
```

The straightforward chain must preserve this logical order. More processors
cannot make `z` use a `y` that does not yet exist.

### 4.2 Independent work

Compare:

```text
p = 10 + 5
q = 20 × 3
```

Neither calculation requires the other's result.

```mermaid
flowchart LR
    P[p = 10 + 5]
    Q[q = 20 × 3]
```

They may execute sequentially or in parallel. Both schedules are valid if there
are no hidden side effects.

### 4.3 A dependency graph

Programs can be represented as a directed acyclic graph for a region of work:

```text
a = x + y
b = p × q
c = a - b
d = c × 2
```

```mermaid
flowchart TD
    A[a = x + y] --> C[c = a - b]
    B[b = p × q] --> C
    C --> D[d = c × 2]
```

`a` and `b` may run in parallel. `c` waits for both. `d` waits for `c`.

This graph contains parallel work and sequential stages. Most real programs are
mixtures rather than purely sequential or purely parallel.

### 4.4 Reduction: independent work followed by combination

Suppose we want the sum of eight numbers.

A sequential method forms a chain:

```text
(((((((a+b)+c)+d)+e)+f)+g)+h)
```

A tree method first forms independent pairs:

```mermaid
flowchart BT
    A["a"] --> AB["a + b"]
    B[b] --> AB
    C["c"] --> CD["c + d"]
    D[d] --> CD
    E["e"] --> EF["e + f"]
    F[f] --> EF
    G["g"] --> GH["g + h"]
    H[h] --> GH
    AB --> L["(a + b) + (c + d)"]
    CD --> L
    EF --> R["(e + f) + (g + h)"]
    GH --> R
    L --> S["final sum"]
    R --> S
```

Each level has independent work, but levels depend on earlier levels. Parallel
algorithms often restructure work to expose this kind of tree.

### 4.5 Scheduling in waves

Available parallelism may exceed physical resources.

```text
Ready operations:  [1][2][3][4][5][6][7][8]
Execution slots:   [A][B]

Wave 1: operations 1 and 2
Wave 2: operations 3 and 4
Wave 3: operations 5 and 6
Wave 4: operations 7 and 8
```

The operations are independent, but only two execute simultaneously. GPUs use
large-scale scheduling to process far more ready threads than can issue an
instruction in one instant.

### 4.6 Data parallelism

**Data parallelism** applies the same broad operation to different data items.
Elementwise list addition is a simple example:

```text
Operation rule: C[i] = A[i] + B[i]
Different data: i = 0, 1, 2, 3, ...
```

Image filters, tensor elementwise functions, and many matrix operations expose
data parallelism.

### 4.7 Task parallelism

**Task parallelism** executes different kinds of independent work:

```text
Task A: tokenize one request
Task B: write logs
Task C: prepare another response
```

CPU servers frequently exploit task parallelism. GPUs are especially designed
for large groups of similarly structured data-parallel work, although they can
execute diverse kernels over time.

### 4.8 Parallel fraction limits total speedup

If part of a task remains sequential, accelerating only the parallel part has a
limit.

Example:

```text
Original total time:      100 ms
Sequential preparation:   40 ms
Parallel calculation:     60 ms
```

Even if the 60 ms calculation became infinitely fast, the total could not drop
below the remaining 40 ms preparation in this simplified scenario.

This is the intuition behind Amdahl's law. You do not need its formula yet. The
lesson is:

> End-to-end speedup is limited by work that the optimization does not improve.

### Section 4 checkpoint

Given a sequence of operations, draw arrows only where a later result truly
requires an earlier result. Identify which operations could be ready together
and which must wait.

---

## 5. CPU Design Priorities

### 5.1 What a CPU must handle

A general-purpose CPU runs:

- Operating-system code
- Browser and application logic
- File and network handling
- Database queries
- Python interpreters
- Tokenization and request orchestration
- Small and irregular calculations

These workloads often include unpredictable decisions and limited parallelism.
The CPU is designed to make a few complicated instruction streams progress
quickly.

### 5.2 CPU cores

A **CPU core** is a sophisticated execution engine capable of running its own
instruction stream. A multicore CPU has several such cores.

```text
Conceptual CPU
┌──────────────────────────────────────────────────┐
│ Core 0 │ Core 1 │ Core 2 │ Core 3 │ Shared cache │
└──────────────────────────────────────────────────┘
```

The diagram is not a physical floor plan. Actual CPUs vary significantly.

### 5.3 Branch prediction

Programs frequently make decisions:

```text
if request_is_valid:
    process_request
else:
    return_error
```

The processor may not know which path is needed until a comparison finishes.
A **branch predictor** guesses the likely path so the CPU can begin preparing
work instead of waiting. A correct prediction saves time. A wrong prediction
requires discarding incorrectly prepared work and restarting on the correct
path.

```mermaid
flowchart TD
    B[Branch encountered] --> P[Predict path]
    P --> S[Speculatively prepare instructions]
    S --> C{Prediction correct?}
    C -- Yes --> K[Keep progress]
    C -- No --> D[Discard wrong-path work and recover]
```

This hardware is valuable for irregular control flow but consumes design area
and power.

### 5.4 Out-of-order execution

Suppose instruction A waits for memory while independent instruction B is ready.
An out-of-order CPU may execute B first, even if A appeared earlier in program
order, while still preserving the program's observable correctness.

```text
Program order:       A(waiting), B(ready), C(depends on A)
Possible execution:  B, wait/other work, A, C
```

This dynamically discovers instruction-level parallelism in a single thread.

### 5.5 Caches and low-latency focus

CPUs devote substantial resources to multi-level caches that keep likely-needed
instructions and data close to each core. They also use complex scheduling,
prediction, and speculative machinery.

Simplified design-priority picture:

```text
CPU die — conceptual only
┌────────────────────────────────────────────────────┐
│ Sophisticated cores                                │
│ Large cache hierarchy                              │
│ Branch prediction and out-of-order scheduling      │
│ Interfaces to memory and devices                   │
└────────────────────────────────────────────────────┘
```

The priority is not “do little work.” It is:

> Make a relatively small number of diverse, dependency-heavy instruction
> streams advance with low latency.

### 5.6 Why CPUs are still important in inference

In an inference system, the CPU may:

- Receive network requests
- Validate parameters
- Tokenize text
- Allocate and prepare inputs
- Run Python and server logic
- Submit GPU operations
- Select output tokens
- Serialize and return responses
- Record metrics and logs

The GPU accelerates suitable numerical work. It does not eliminate the rest of
the system.

### Section 5 checkpoint

Explain why branch prediction and out-of-order execution are useful for a CPU
running irregular application logic, and why those capabilities represent a
different design investment from maximizing parallel arithmetic throughput.

---

## 6. GPU Design Priorities

### 6.1 Terms Needed Before We Discuss GPU Inference

This section uses several machine-learning and NVIDIA terms that have not yet
been introduced. Define them first; do not infer their meaning from their names.

#### Token

A language model does not normally receive complete English words directly. A
**token** is one unit produced by a model's tokenizer. Depending on the
tokenizer and text, a token can represent:

- A whole word
- Part of a word
- Punctuation
- Whitespace combined with nearby text
- A byte or character-like unit

The exact split belongs to the selected tokenizer. For example, one tokenizer
might split:

```text
"Why is the sky blue?"
```

into the illustrative pieces:

```text
["Why", " is", " the", " sky", " blue", "?"]
```

Another tokenizer may split the same text differently.

#### Token ID

The tokenizer owns a vocabulary: a mapping between token pieces and integer
identifiers. A **token ID** is the integer the model uses to identify a token.

```text
Token piece       Hypothetical ID
-----------       ---------------
"Why"                  812
" is"                   27
" the"                 279
" sky"              13,180
" blue"              6,437
"?"                     30
```

These numbers are intentionally hypothetical. Token IDs have meaning only in
the vocabulary of a particular tokenizer and model.

#### Tensor

A **tensor** is a multidimensional collection of numerical values together with
properties such as shape and data type. At this point, think of a tensor as a
generalization of a number, list, or matrix:

```text
One number:                 scalar tensor
One-dimensional list:       vector tensor
Two-dimensional grid:       matrix tensor
Three or more dimensions:   higher-dimensional tensor
```

Section 8 develops tensors in more detail. The definition appears here because
GPU frameworks package model inputs, weights, and intermediate results as
tensors.

#### Embedding

A token ID is only a category label; arithmetic on the number `812` would not
capture the meaning of “Why.” An **embedding** maps each token ID to a learned
vector of floating-point values.

```mermaid
flowchart LR
    PIECE["Token piece: Why"] --> ID["Token ID: 812"]
    ID --> LOOKUP["Embedding-table row lookup"]
    LOOKUP --> VECTOR["Learned vector: 0.2, -0.1, 0.7, 0.4, ..."]
```

The real vector can contain thousands of values. The four-value vector above is
only a teaching illustration.

#### Model weights

**Weights** are numerical parameters learned during training. During inference,
the model repeatedly uses large weight tensors to transform input and
intermediate tensors. Weight matrices encode learned transformations; they are
not a database of human-readable sentences.

#### CUDA

**CUDA** is NVIDIA's software platform and programming model for using NVIDIA
GPUs for general-purpose computation. CUDA supplies concepts and interfaces for
activities such as:

- Selecting a GPU
- Allocating GPU memory
- Moving data
- Submitting functions for GPU execution
- Ordering and synchronizing work
- Using optimized numerical libraries

CUDA is not the GPU itself, and it is not another name for a Tensor Core.
Frameworks such as PyTorch use CUDA-enabled libraries and runtime interfaces so
the application can ask an NVIDIA GPU to execute suitable operations. Lesson 06
explains the CUDA software stack fully; this definition is enough for the
request walkthrough below.

#### GPU kernel

A **kernel** is a function submitted for parallel execution on the GPU. A
framework operation may launch one kernel, several kernels, or a fused kernel
representing several conceptual operations.

#### Streaming Multiprocessor

A **Streaming Multiprocessor (SM)** is a major GPU processing unit that contains
thread schedulers, registers, shared memory/cache resources, and several kinds
of execution pipelines. A GPU contains multiple SMs. They collectively execute
thread groups assigned by GPU hardware.

Lesson 02 explains kernels, threads, blocks, warps, and SM scheduling in depth.

### 6.2 Origins: From Graphics to General-Purpose Computation

Graphics workloads repeatedly apply related calculations to many vertices,
fragments, and pixels. For example, millions of output pixels may need colors
computed from geometry, textures, lighting, and viewing information. That work
contains large amounts of structured data parallelism.

GPU designs evolved around this need for high aggregate arithmetic throughput.
As GPU programmability increased, engineers used the same broad architecture
for non-graphics workloads with similar parallel structure, including:

- Physical simulation
- Scientific computing
- Image and video processing
- Machine learning
- Transformer inference

CUDA is the NVIDIA platform that exposes NVIDIA GPUs for this
general-purpose-computing use. The important chain is:

```mermaid
flowchart LR
    GRAPHICS["Graphics requires similar calculations over many data items"]
    GRAPHICS --> ARCH["GPU develops throughput-oriented parallel architecture"]
    ARCH --> CUDA["CUDA exposes that architecture for general computation"]
    CUDA --> ML["Frameworks use CUDA to run suitable machine-learning operations"]
```

### 6.3 The CPU/GPU Design Tradeoff — The Main Rule and the Nuance

The main distinction is exactly the one you suspected:

> **CPUs are primarily optimized for flexible control flow and low-latency
> progress through a modest number of instruction streams. GPUs are primarily
> optimized for high-throughput execution of large amounts of structured
> parallel numerical work.**

The earlier wording—“CPUs perform parallel work; GPUs handle control flow”—was
not intended to reverse those roles. It attempted to say neither capability is
exclusive, but it gave the exception too much emphasis and obscured the rule.

Here is the corrected comparison:

| Design concern | CPU emphasis | GPU emphasis |
| --- | --- | --- |
| Primary strength | Flexible application and control logic | Structured parallel numerical computation |
| Individual execution resources | Fewer sophisticated cores | Many SMs containing many execution pathways |
| Branching | Strong machinery for irregular and unpredictable branches | Supports branches, but divergence within thread groups can waste work |
| Latency strategy | Make one/few threads advance quickly using caches, prediction, and out-of-order execution | Keep many thread groups available and issue other work while some groups wait |
| Memory design emphasis | Large low-latency cache hierarchy | High aggregate device-memory bandwidth plus on-chip reuse |
| Typical inference responsibilities | Networking, validation, tokenization, scheduling, orchestration | Embeddings and large tensor operations across model layers |

#### Why the nuance still matters

“CPU handles control; GPU handles parallel work” is a good first approximation,
not an absolute boundary:

- Modern CPUs have multiple cores and vector instructions, so they can execute
  parallel numerical work.
- GPUs execute instructions containing comparisons, loops, and branches, so
  they do perform control flow.
- A full inference engine may move token selection or generation bookkeeping to
  the GPU to avoid synchronization.
- Some small tensor operations may be faster on the CPU because GPU overhead
  exceeds useful work.

The accurate conclusion is asymmetric:

```text
PRIMARY DESIGN PRIORITY

CPU ───────────────────────────────▶ flexible control + low thread latency
GPU ───────────────────────────────▶ structured parallel throughput

SECONDARY CAPABILITY

CPU can perform parallel math.
GPU can execute control-flow instructions.
```

The secondary capabilities do not erase the primary design priorities.

### 6.4 Frame-by-Frame Walkthrough: One Prompt Reaches the GPU

The following figure functions like six animation frames. Read it from the
upper-left downward, then continue at the upper-right.

![Six-frame prompt-to-GPU inference flow](assets/prompt-to-gpu-inference-flow.svg)

The dimensions, vectors, and token IDs are deliberately tiny or hypothetical
so the transformations can be seen. A real model uses its own tokenizer and
weight tensors with much larger dimensions.

#### Frame 1 — CPU receives and validates the request

Suppose a client sends:

```text
Prompt: "Why is the sky blue?"
Maximum new tokens: 20
Sampling policy: deterministic greedy selection
```

Server code running on the CPU may:

1. Parse the HTTP request.
2. Confirm the prompt is a string.
3. Check prompt and output limits.
4. Confirm the requested model is ready.
5. Choose the appropriate tokenizer and generation configuration.

These are branching and orchestration tasks. They are natural CPU work.

#### Frame 2 — CPU tokenizer converts text to token IDs

Using the hypothetical mapping introduced in Section 6.1:

```text
Text:
"Why is the sky blue?"

Token pieces:
["Why", " is", " the", " sky", " blue", "?"]

Token IDs:
[812, 27, 279, 13180, 6437, 30]
```

The position in the ID list preserves token order:

```text
position:    0     1     2       3      4    5
token ID:  812    27   279   13180   6437   30
```

The model uses both token identity and position. Reordering these IDs changes
the input sequence.

#### Frame 3 — Framework prepares device work

The framework represents the IDs as an integer tensor with a shape such as:

```text
input_ids shape = [batch size, token count]
                = [1, 6]
```

If the tensor is in host memory and the model is on the GPU, the required input
data is made available in GPU memory. The CPU-side framework then submits CUDA
operations. The CPU does not send English words directly to arithmetic units;
it submits operations over encoded tensors.

#### Frame 4 — GPU performs embedding lookup

The embedding table can be imagined as a large matrix:

```text
embedding_table shape = [vocabulary size, hidden size]
```

Each token ID selects one row. For teaching purposes, pretend the hidden size is
only four:

```text
Token       Hypothetical embedding row
-----       --------------------------------
"Why"       [ 0.2, -0.1,  0.7,  0.4]
" is"       [ 0.0,  0.5, -0.3,  0.8]
" the"      [ 0.6,  0.2,  0.1, -0.4]
" sky"      [ 0.9, -0.5,  0.3,  0.2]
" blue"     [ 0.7,  0.1,  0.8, -0.2]
"?"         [-0.1,  0.4,  0.0,  0.5]
```

Stacking the rows produces a token-state matrix `X`:

```text
X shape = [6 token positions, 4 hidden features]

    feature 0  feature 1  feature 2  feature 3
    ---------  ---------  ---------  ---------
Why     0.2       -0.1        0.7        0.4
 is     0.0        0.5       -0.3        0.8
 the    0.6        0.2        0.1       -0.4
 sky    0.9       -0.5        0.3        0.2
 blue   0.7        0.1        0.8       -0.2
 ?     -0.1        0.4        0.0        0.5
```

These values do not individually translate back into simple concepts such as
“question” or “color.” Meaning is distributed across learned dimensions and
transformations.

#### Frame 5 — Transformer layers apply matrix operations

A simplified linear transformation is:

```text
Y = X × W
```

If `X` has shape `[6, 4]` and a weight matrix `W` has shape `[4, 8]`, then:

```text
[6, 4] × [4, 8] → [6, 8]
```

`Y` contains 48 output values. Each output value is a dot product between one
row of `X` and one column of `W`.

The next figure uses even smaller integer matrices—three token rows, two input
features, and four output features—so every output can be verified manually.

![Matrix outputs divided into parallel GPU work](assets/parallel-matrix-output.svg)

For one cell in that figure:

```text
X row 1       = [3, 4]
W column 2    = [2, 1]

Y[1,2] = (3 × 2) + (4 × 1)
       = 6 + 4
       = 10
```

Other output cells use other row/column combinations. Once input values are
available, output cells do not require one another's final values. This creates
available parallel work.

Real GPU matrix kernels do not normally assign one simplistic standalone
thread to each entire dot product. Efficient libraries divide matrices into
**tiles**, assign tiles to thread blocks, cooperate across thread groups, reuse
data, and use specialized instructions. The correct beginner insight is:

```text
Large output matrix
        ↓ divide into tiles
Many independent or regularly structured output regions
        ↓ schedule blocks across finite SMs
Many arithmetic operations execute concurrently and in waves
```

Transformer blocks repeat several large learned transformations. In attention,
the model forms query, key, and value tensors using operations commonly written
as:

```text
Q = X × WQ
K = X × WK
V = X × WV
```

The model then calculates attention relationships and applies additional
projections and feed-forward layers. Section 8 introduces that flow; Module 03
develops the mechanics fully.

#### Frame 6 — Final projection produces next-token scores

After all transformer layers, a final transformation produces one score for
each token in the model vocabulary. These scores are called **logits**.

Tiny illustration:

```text
Candidate token      Logit
---------------      -----
"Because"             8.2
"The"                 6.9
"Blue"                3.1
"Car"                -1.4
```

A decoding policy converts logits into a next-token choice. If the chosen token
is `"Because"`, the generated sequence becomes conceptually:

```text
"Why is the sky blue?" + "Because"
```

The model then performs another decode iteration to choose the following token.
The generated answer emerges one selected token at a time—not as a complete
sentence stored inside one matrix.

### 6.5 CPU and GPU Activity Over Time

The following sequence diagram emphasizes responsibility and ordering:

```mermaid
sequenceDiagram
    participant Client
    participant CPU as CPU service and framework
    participant Driver as CUDA and NVIDIA driver
    participant GPU
    Client->>CPU: Prompt and generation settings
    CPU->>CPU: Validate request
    CPU->>CPU: Tokenize text into IDs
    CPU->>CPU: Build input tensor and generation state
    CPU->>Driver: Submit transfer and model operations
    Driver->>GPU: Queue kernels and data movement
    GPU->>GPU: Embedding lookup
    GPU->>GPU: Parallel transformer tensor operations
    GPU->>GPU: Final projection to logits
    GPU-->>CPU: Make required result or selected ID available
    CPU->>CPU: Decode ID to text and manage response
    CPU-->>Client: Generated output
```

Implementations differ. Some keep token selection and more generation logic on
the GPU to avoid repeated CPU synchronization. The diagram communicates the
conceptual responsibilities, not a mandatory transfer after every token.

### 6.6 Throughput-Oriented GPU Resources

A GPU allocates substantial resources to:

- Many arithmetic execution pathways
- Scheduling many active thread groups
- High aggregate device-memory bandwidth
- Specialized matrix multiply-accumulate hardware
- Hiding latency by switching among ready work

```text
GPU die — conceptual, not to scale
┌──────────────────────────────────────────────────────┐
│ SM │ SM │ SM │ SM │ SM │ SM │ SM │ SM │ ...       │
│                                                      │
│          Shared L2 cache and interconnect            │
│                                                      │
│              Memory controllers                      │
└──────────────────────────────────────────────────────┘
```

The GPU still has finite resources. A matrix with millions of logical output
calculations is divided and scheduled across available hardware over time.

### 6.7 GPU Threads Are Logical Workers, Not Permanent Cores

A GPU program may define thousands or millions of logical threads. A thread
describes one instance of work, such as “help calculate this output region.” It
does not mean the GPU contains one permanent physical core for every thread.

```text
Logical threads:  [0][1][2][3][4][5] ... [999999]
Physical GPU:      finite SMs and execution pipelines
Execution:         logical work scheduled in groups and waves
```

Lesson 02 explains how threads are grouped into blocks and warps. For now,
separate the **quantity of described work** from the **quantity of physical
hardware available at one instant**.

### 6.8 Hiding Latency With Many Ready Groups

When one thread group waits for data, an SM can issue eligible instructions
from another ready group.

```mermaid
sequenceDiagram
    participant W1 as Thread group 1
    participant SM as SM scheduler
    participant W2 as Thread group 2
    W1->>SM: Issue memory request and become ineligible
    SM->>W2: Issue ready arithmetic instruction
    W1-->>SM: Requested data becomes available
    SM->>W1: Resume eligible instruction
```

This does not shorten the memory access itself. It tries to keep execution
resources productive during the wait. It requires enough independent ready
groups; a tiny workload may leave much of the GPU unused.

### 6.9 Host and Device Cooperation

In CUDA terminology:

- The **host** is the CPU-side system.
- The **device** is the GPU.

```mermaid
flowchart LR
    subgraph Host
      APP["Application and PyTorch"]
      RAM["System memory"]
    end
    subgraph Device
      GPU["GPU execution"]
      VRAM["GPU device memory"]
    end
    APP -- "submit CUDA work" --> GPU
    RAM <-- "transfer data when required" --> VRAM
```

The host prepares and orchestrates work. The device performs submitted GPU
operations. Data movement and synchronization are real costs, so efficient
systems avoid unnecessary back-and-forth transfers.

### 6.10 Why Large Tensor Operations Help

Now that **tensor** has been defined, we can state the connection precisely.
If a tensor operation contains millions of independent or regularly structured
output calculations, the GPU has a large pool of ready work. That allows it to:

- Divide output regions into blocks or tiles
- Distribute blocks across many SMs
- Keep multiple thread groups ready
- Hide some memory latency
- Reuse data within faster parts of the memory hierarchy
- Use specialized matrix hardware when the operation, shape, dtype, and kernel
  are eligible

A four-element list is useful for understanding independence but is too small
to justify a real GPU launch by itself. Transformer layers apply the same broad
principle at enormous scale.

### Section 6 Checkpoint

Without looking back, explain:

1. CUDA, token, token ID, tensor, embedding, weight, kernel, and SM.
2. The primary CPU/GPU design distinction and the non-exclusive nuance.
3. The path from `"Why is the sky blue?"` to token IDs and an embedding matrix.
4. How `Y = X × W` creates many output calculations that GPU kernels can divide
   into tiles and schedule across SMs.
5. Why logical GPU threads are not permanent physical cores.
6. Why a large supply of ready work can hide memory latency.
7. Why a real implementation may keep token selection on the GPU rather than
   copying a result to the CPU after every iteration.

---

## 7. Matrix Multiplication, Step by Step

### 7.1 Scalars, vectors, and matrices

- A **scalar** is one value: `7`.
- A **vector** is an ordered one-dimensional collection: `[2, 4, 6]`.
- A **matrix** is a rectangular two-dimensional collection.

```text
Matrix A with 2 rows and 3 columns:

┌ 1  2  3 ┐
└ 4  5  6 ┘

Shape: 2 × 3
```

The shape describes the number of entries along each dimension.

### 7.2 The multiplication shape rule

If:

```text
A has shape M × K
B has shape K × N
```

then:

```text
C = A × B has shape M × N
```

The inner dimension `K` must match because each output uses a row of length `K`
from A and a column of length `K` from B.

```text
(M × K) × (K × N) → (M × N)
       matching K
```

### 7.3 Complete 2 × 2 example

```text
A = ┌ 1  2 ┐       B = ┌ 5  6 ┐
    └ 3  4 ┘           └ 7  8 ┘

C = A × B
```

Each output is a row-column dot product:

```text
C[0,0] = A row 0 · B column 0
       = (1 × 5) + (2 × 7)
       = 5 + 14
       = 19

C[0,1] = A row 0 · B column 1
       = (1 × 6) + (2 × 8)
       = 6 + 16
       = 22

C[1,0] = A row 1 · B column 0
       = (3 × 5) + (4 × 7)
       = 15 + 28
       = 43

C[1,1] = A row 1 · B column 1
       = (3 × 6) + (4 × 8)
       = 18 + 32
       = 50
```

Result:

```text
C = ┌ 19  22 ┐
    └ 43  50 ┘
```

### 7.4 Dependencies within one output

For `C[0,0]`, the two products can be computed independently:

```mermaid
flowchart TD
    P1[1 × 5 = 5] --> S[5 + 14 = 19]
    P2[2 × 7 = 14] --> S
```

The final sum must wait for both products. With a longer dot product, partial
sums form a reduction. The calculation contains parallel multiplication and a
dependent accumulation structure.

Hardware often uses fused multiply-accumulate instructions and tiled algorithms
rather than literally creating this exact graph, but the data dependencies are
the same.

### 7.5 Independence across output cells

Once A and B are available, each output cell can be computed without needing
another output cell:

```mermaid
flowchart TD
    IN[Input matrices A and B]
    IN --> C00[Compute C00]
    IN --> C01[Compute C01]
    IN --> C10[Compute C10]
    IN --> C11[Compute C11]
    C00 --> OUT[Completed C]
    C01 --> OUT
    C10 --> OUT
    C11 --> OUT
```

The final matrix is complete only after all output cells are ready, but their
calculations expose parallelism.

### 7.6 Scaling the example

If C has shape 4,096 × 4,096, it contains:

```text
4,096 × 4,096 = 16,777,216 output elements
```

If each output is a dot product of length 4,096, there is an enormous amount of
multiply-accumulate work. Efficient GPU libraries divide matrices into tiles,
reuse input values through the memory hierarchy, and distribute work across
SMs.

```text
Large matrices
      ↓ divide into tiles
Thread blocks process output tiles
      ↓ distributed across SMs
Many multiply-accumulate operations
      ↓
Completed output matrix
```

Lesson 03 explains general arithmetic pipelines and Tensor Cores. Lesson 04
explains why tiling and reuse reduce expensive data movement.

### 7.7 Matrix multiplication is parallel but not order-free

It would be incorrect to say “all matrix operations happen in any order.”
Instead:

- Output cells are broadly independent from one another.
- Products within a dot product can be formed independently.
- Products must be combined to produce the output.
- Input data must be available before it is used.
- The final consumer must wait for required outputs.
- Floating-point regrouping can introduce small numerical differences because
  finite-precision addition is not perfectly associative.

Parallel algorithms preserve required dependencies while choosing an efficient
valid schedule.

### Section 7 checkpoint

For one 2 × 2 result cell, identify the loads, multiplications, additions, and
store. Then explain which work can overlap across the four result cells.

---

## 8. Why Transformer Inference Contains GPU-Friendly Work

### 8.1 What is inference?

**Inference** means using an already-trained model to produce outputs from new
inputs. For a decoder-only language model, inference repeatedly predicts a next
token based on the tokens seen so far.

Training adjusts model weights. Inference primarily reads those weights and
uses them in numerical operations.

### 8.2 What is a tensor?

A **tensor** is a multidimensional collection of values with properties such as:

- **Shape:** number of positions along each dimension
- **Data type:** representation used for each value
- **Device:** where its storage resides
- **Layout or strides:** how logical positions map to memory

More concretely, a tensor is a regular collection of same-data-type values that
can be addressed with integer indices, plus metadata describing how to
interpret and locate those values. A rank-3 tensor is not necessarily stored as
a literal geometric cube. Its axes are a logical indexing organization; its
underlying storage is commonly a one-dimensional region of memory.

Examples:

```text
Scalar: 7                              shape: []
Vector: [2, 4, 6]                      shape: [3]
Matrix: [[1, 2], [3, 4]]               shape: [2, 2]
Token states                           shape: [batch, tokens, hidden_size]
```

If a token-state tensor has shape `[2, 128, 4096]`:

- `2` means two sequences in the batch.
- `128` means 128 token positions per sequence.
- `4096` means each token position is represented by 4,096 numerical features.

Total values:

```text
2 × 128 × 4096 = 1,048,576 values
```

The word **dimension** is overloaded, so we will be precise:

- An **axis** is one direction in which a tensor can be indexed.
- The **size of an axis** is the number of positions along that axis.
- The **rank** is the number of axes. Rank is not the number of stored values.
- The **shape** lists the axis sizes in order.

The following figure builds rank one axis at a time. Every colored square is
one scalar value. Brackets in a shape describe axes; they are not matrix rows.

![Scalar, vector, matrix, and rank-3 tensor with every axis labeled](assets/tensor-rank-and-shape.svg)

#### Reading an index

For the matrix below, the first index chooses a row and the second chooses a
column:

```text
                 column 0   column 1   column 2
                            axis 1 →
row 0, axis 0 ↓      10         11         12
row 1                20         21         22

shape = [2 rows, 3 columns]
matrix[1, 2] = 22
       │  └──────── choose column 2
       └─────────── choose row 1
```

A rank-3 tensor adds another index. It is often easiest to understand it as a
stack of matrices. `T[batch, token, feature]` means:

1. Choose a sequence from the batch.
2. Choose a token position in that sequence.
3. Choose one numerical feature belonging to that token position.

#### Expanding `[2, 128, 4096]`

![A two-sequence token-state tensor with token and hidden-feature axes expanded](assets/token-state-tensor-anatomy.svg)

The picture deliberately cannot draw all 1,048,576 cells. Ellipses mean
“positions omitted from the drawing,” not missing data. The tensor contains:

```text
2 sequences
× 128 token positions in each sequence
× 4,096 feature values for each token position
= 1,048,576 scalar values
```

One particular scalar might be written:

```text
H[1, 5, 37]
  │  │   └── feature 37
  │  └────── token position 5
  └───────── sequence 1 (the second sequence because indexing starts at 0)
```

The 4,096 values at `H[1, 5, :]` collectively represent the model's current
state for that token position. Individual features generally do not have a
simple fixed translation such as “feature 37 means blue.” Meaning is distributed
across the vector and transformed from layer to layer.

#### Shape is only part of a tensor's description

Two tensors can have the same shape and still differ:

| Property | Example | What it answers |
| --- | --- | --- |
| Shape | `[2, 128, 4096]` | How many positions exist along each axis? |
| Data type | `float16` | How is each value encoded, and how many bytes does it use? |
| Device | `cuda:0` | In which processor's attached memory does the storage reside? |
| Stride | `[524288, 4096, 1]` | How far in storage must we move when an index increases by one? |

For a contiguous row-major `[2, 128, 4096]` tensor, adjacent features are next
to one another. Moving forward one token skips 4,096 stored values; moving
forward one batch item skips `128 × 4096 = 524,288` values. Lesson 04 develops
layout and memory consequences.

The data type determines bytes per value. For example, if every value occupies
2 bytes, the raw storage for this tensor is:

```text
1,048,576 values × 2 bytes/value = 2,097,152 bytes = 2 MiB
```

Lesson 04 turns this reasoning into broader memory estimates.

### 8.3 Token embeddings

Text is first converted into token IDs. An embedding lookup maps each token ID
to a learned vector.

```mermaid
flowchart LR
    T[Text] --> IDS[Token IDs]
    IDS --> E[Embedding lookup]
    E --> V[Vector for each token]
```

The model does not directly calculate on English words. It calculates on
numerical tensors.

### 8.4 Linear transformations

A transformer repeatedly applies learned weight matrices to token-state
vectors. In simplified form:

```text
output = input × weights + bias
```

For many token positions and a batch, these become large matrix operations.

Example shapes:

```text
Input token states:  [256 token positions, 4096 features]
Weight matrix:       [4096 input features, 4096 output features]
Output states:       [256 token positions, 4096 output features]
```

Before using the large numbers, examine a complete teaching example:

![Fully labeled token-state by weight-matrix multiplication](assets/linear-transformation-from-tokens.svg)

Here `X` contains three token positions with two input features each. `W`
describes how two input features contribute to four output features. Therefore:

```text
X shape: [3 token positions, 2 input features]
W shape: [2 input features, 4 output features]
Y shape: [3 token positions, 4 output features]

             shared dimension
                    ┌─┴─┐
[3 tokens, 2 input] × [2 input, 4 output] → [3 tokens, 4 output]
 └ output rows ┘                         └── output columns ──┘
```

For example, output `Y[token 1, output feature 2]` uses row 1 of `X` and
column 2 of `W`:

```text
X row 1 = [3, 4]
W column 2 = [2, 1]
Y[1, 2] = (3 × 2) + (4 × 1) = 10
```

The same shape rule applies to the realistic example:

```text
[256 token positions, 4096 input features]
× [4096 input features, 4096 output features]
→ [256 token positions, 4096 output features]
```

That output contains:

```text
256 × 4096 = 1,048,576 output values
```

Each output is a dot product of length 4,096, so forming all outputs requires
approximately:

```text
1,048,576 outputs × 4,096 multiply-accumulate steps per output
= 4,294,967,296 multiply-accumulate steps
```

This count describes mathematical work, not elapsed time or the number of
physical GPU cores. Many output cells and tiles can be worked on concurrently;
finite hardware schedules them in waves. The bias, if present, contributes one
additional value to each output cell after the dot product.

### 8.5 Attention at a first-pass level

Attention creates query, key, and value tensors through linear transformations,
calculates relationships between token positions, combines information, and
projects the result.

```mermaid
flowchart LR
    X[Token states] --> Q[Query projection]
    X --> K[Key projection]
    X --> V[Value projection]
    Q --> SCORE[Attention scores]
    K --> SCORE
    SCORE --> MIX[Weighted combination]
    V --> MIX
    MIX --> O[Output projection]
```

This is a structural preview, not a complete attention lesson. Module 03 will
explain causal masking and the KV cache.

### 8.6 Feed-forward layers

Transformer blocks also contain feed-forward networks, often with an expansion
to a larger intermediate dimension followed by projection back to the hidden
dimension. These are dominated by large linear operations plus elementwise
activation functions.

```mermaid
flowchart LR
    H[Hidden states] --> UP[Large up-projection]
    UP --> ACT[Elementwise activation]
    ACT --> DOWN[Down-projection]
    DOWN --> R[Output states]
```

### 8.7 Not every transformer operation is equally GPU-friendly

Different operations have different characteristics:

- Large matrix multiplication can expose high parallelism and data reuse.
- Elementwise operations expose parallelism but may perform little math per
  byte moved.
- Normalization requires reductions and elementwise work.
- Token selection and Python control flow can be small or CPU-oriented.
- One-token decode creates smaller operation shapes than processing many prompt
  tokens together.

Therefore “transformers use matrix multiplication” is only the beginning of a
performance explanation.

### 8.8 The CPU-GPU pipeline

A simplified inference request crosses both processors:

```mermaid
sequenceDiagram
    participant Client
    participant CPU
    participant GPU
    Client->>CPU: Send prompt
    CPU->>CPU: Validate and tokenize
    CPU->>GPU: Submit tensor operations
    GPU->>GPU: Execute model kernels
    GPU-->>CPU: Return required results
    CPU->>CPU: Select/decode token and manage request
    CPU-->>Client: Return output
```

Frameworks may keep much of the generation loop and token selection on device,
and implementations vary. The key point is that end-to-end inference includes
more than GPU arithmetic.

### Section 8 checkpoint

Explain how text becomes numerical tensor work and identify at least three
different operation types in a transformer block.

---

## 9. Prefill and Decode: A Careful Preview

This section provides only the performance foundation. Module 03 develops the
full mechanics.

### 9.0 Start With Time: Which Tokens Exist Right Now?

Before defining prefill, separate two categories that the earlier version called
“known” and “unknown” without enough explanation.

Suppose the user submits this prompt:

```text
"The sky is blue"
```

After tokenization, assume the prompt contains four token pieces:

```text
position       0        1       2        3
token        "The"    " sky"  " is"   " blue"
```

These are **known tokens** because their identities are already present in the
request. The server does not need the model to guess them. It can immediately
convert all four to token IDs and supply those IDs to the model.

The token that should follow `" blue"` is **not yet known**. The request did not
contain it, and the model has not selected it yet. We can draw the boundary at
the instant the request arrives:

```text
ALREADY PROVIDED BY THE USER                  NOT SELECTED YET

position 0   position 1   position 2   position 3   position 4
  "The"       " sky"        " is"       " blue"         ?
└──────────────── prompt tokens ────────────────┘    next token
         known to the program now                     unknown now
```

“Known” does **not** mean that the model understands the token, that its answer
is predetermined, or that the token came from training data. It means only:

> **At this moment in this request, the program already has this token's ID.**

After the model selects a token for position 4—suppose it selects
`" because"`—that token becomes known and is appended to the sequence. Position
5 is then the new unknown:

```text
[The] [ sky] [ is] [ blue] [ because] [?]
                            newly known  next unknown
```

This boundary moves one position to the right after every generated token.

![Known prompt tokens, the moving unknown boundary, prefill, and repeated decode steps](assets/known-tokens-prefill-decode-timeline.svg)

### 9.1 The Two Stages of Generation

Generation divides naturally at that boundary:

| Stage | Input available at the start | What the stage accomplishes |
| --- | --- | --- |
| **Prefill** | All tokens supplied in the prompt | Processes the prompt, saves reusable attention information, and produces scores for the first token after the prompt. |
| **Decode** | The prompt plus tokens generated so far | Selects one additional token, saves its reusable attention information, and repeats. |

For our example:

```text
PREFILL
input:   [The] [ sky] [ is] [ blue]
output:  scores for position 4 + saved attention information for positions 0–3

DECODE ITERATION 1
select:  [ because] for position 4
output:  scores for position 5 + add position 4 to saved attention information

DECODE ITERATION 2
select:  perhaps [ light] for position 5
output:  scores for position 6 + add position 5 to saved attention information
```

The model does not secretly calculate the whole answer during prefill. It
produces scores for one next-token decision. A decoding policy selects one
token from those scores. Only then does the next decode iteration have its full
input.

### 9.2 Prefill, Slowly and Concretely

**Prefill is the model's first processing pass over the prompt.** The word
“fill” refers partly to filling the KV cache with reusable information for the
prompt positions.

#### Step 1 — The whole prompt becomes rows of numbers

The four known token IDs are mapped to vectors. At the entrance to a layer,
think of one row per prompt position:

```text
Xprompt
shape: [4 token positions, hidden_size]

row 0 → current numerical state for "The"
row 1 → current numerical state for " sky"
row 2 → current numerical state for " is"
row 3 → current numerical state for " blue"
```

All four rows exist before GPU model execution starts because all four token
IDs came from the request.

#### Step 2 — Each attention layer creates Q, K, and V rows

At one attention layer, learned linear transformations create three different
views of every row:

```text
Q = Xprompt × WQ      one query row per prompt position
K = Xprompt × WK      one key row per prompt position
V = Xprompt × WV      one value row per prompt position
```

At a conceptual level:

- A **query** describes what the position is looking for.
- A **key** describes what a position can be matched on.
- A **value** contains information that can be mixed into another position.

Q, K, and V are numerical vectors. They are not English questions, dictionary
keys, or copies of the token text.

#### Step 3 — Queries compare with keys

The model calculates an attention score for query position `i` and key position
`j`. Before masking, the GPU can calculate a full grid of comparisons:

```text
                         KEY POSITION j
                     The    sky     is    blue
QUERY POSITION i
The                  score  score  score  score
sky                  score  score  score  score
is                   score  score  score  score
blue                 score  score  score  score
```

Each row asks: “While updating this query position, how strongly should each key
position matter?”

#### Step 4 — The causal mask blocks information from the future

A decoder-only language model is trained to predict text from left to right.
When it constructs the representation at position 1, that position must not
peek at positions 2 or 3. Otherwise training would leak the very future tokens
the model is supposed to learn to predict.

The **causal mask** is the rule enforcing that restriction:

```text
                         KEY POSITION BEING READ
                         0      1      2      3
QUERY position 0        allow  block  block  block
QUERY position 1        allow  allow  block  block
QUERY position 2        allow  allow  allow  block
QUERY position 3        allow  allow  allow  allow
```

Another way to read it:

```text
position 0 may read: [0]
position 1 may read: [0, 1]
position 2 may read: [0, 1, 2]
position 3 may read: [0, 1, 2, 3]
```

“Future” here means a larger sequence position **relative to the query row**.
Although the server already possesses every prompt token, the mathematical
information-flow rule still prevents an earlier row from using later rows.

This resolves an apparent contradiction:

- All prompt token IDs are known to the server, so their matrix rows can be
  prepared and calculated together.
- Each row is allowed to use only itself and earlier positions, so the causal
  mask blocks forbidden score cells.

The mask changes which calculated scores may influence the output. It does not
require the GPU to process the four prompt tokens using four complete model
passes.

![Prefill matrices with tokens, Q K transpose scores, causal mask, and contextual outputs labeled](assets/prefill-causal-attention-matrices.svg)

#### Step 5 — Allowed scores mix value rows

After masking and normalization, the allowed scores become weights used to
combine value rows. For the last prompt position, a hypothetical result might
conceptually resemble:

```text
updated information for "blue"
= 0.05 × V("The")
+ 0.60 × V(" sky")
+ 0.10 × V(" is")
+ 0.25 × V(" blue")
```

The numbers are illustrative. The result is a contextual numerical vector: the
state for `" blue"` can now contain information gathered from earlier prompt
positions.

#### Step 6 — Prefill saves K and V, then scores the first new token

At every relevant transformer layer, the model retains the key and value rows
created for the prompt. This saved collection is the **KV cache**:

```text
K cache after prefill                 V cache after prefill

position 0: K("The")                  position 0: V("The")
position 1: K(" sky")                 position 1: V(" sky")
position 2: K(" is")                  position 2: V(" is")
position 3: K(" blue")                position 3: V(" blue")
```

The cache does not contain the model's answer. It does not replace the model's
weights. It stores intermediate numerical results that future decode iterations
will need again.

Finally, the last prompt position's state flows through the remaining model
operations and produces logits—scores over possible tokens for position 4. If
the selection policy chooses `" because"`, prefill is complete.

### 9.3 Decode and the KV Cache, Slowly and Concretely

At the start of the first decode iteration, `" because"` has now been selected.
Its ID is therefore known and can be processed as the newest input position.

#### What would happen without a KV cache?

Attention for the new position needs keys and values representing the earlier
positions. Without saved results, the system would repeatedly recreate the old
K and V rows:

```text
iteration 1: recreate K/V for positions 0–3, then process position 4
iteration 2: recreate K/V for positions 0–4, then process position 5
iteration 3: recreate K/V for positions 0–5, then process position 6
```

That repeats work whose inputs and results have not changed.

#### What happens with a KV cache?

The old rows are retrieved from device memory. Only the newest position needs
new K and V rows:

```text
REUSE from cache                         CALCULATE now

K0 K1 K2 K3                              K4 for " because"
V0 V1 V2 V3                              V4 for " because"
```

The model also calculates a new query `Q4`. That query compares with all keys
available through position 4:

```text
Q4 compares with [K0, K1, K2, K3, K4]
                         │
                         ▼
             five attention scores
                         │
                         ▼
scores mix [V0, V1, V2, V3, V4]
                         │
                         ▼
        context for newest position 4
```

Position 4 can read all positions 0–4 because all are at or before its own
position. There is no future prompt row to mask for this single newest query.

![Decode matrix growth showing a new query row and cached key-value rows](assets/decode-kv-cache-matrix-growth.svg)

After the layer uses the rows, `K4` and `V4` are appended to the cache:

```text
before: cache covers positions [0, 1, 2, 3]
after:  cache covers positions [0, 1, 2, 3, 4]
```

The model eventually produces logits for position 5 and selects another token.
That selected token becomes the input to the next iteration.

#### What the cache saves—and what it costs

The KV cache saves **recomputation of old key and value projections**. It does
not make attention free:

- The new token still passes through every transformer layer.
- A new Q, K, and V must be formed for the new position.
- The new query still compares with the growing collection of cached keys.
- The attention result still combines cached values.
- New K/V rows consume additional GPU memory at every relevant layer.

Therefore the cache trades memory capacity for less repeated computation. Its
memory usage grows with sequence length.

#### Check your mental model

For the prompt `[The, sky, is, blue]`, answer these before continuing:

1. Why are the four prompt tokens called known before prefill?
2. Which token is unknown at that moment?
3. Why may query position 1 not read key position 3 during prefill?
4. Does the causal mask mean the GPU must run four complete sequential model
   passes? Why not?
5. What numerical rows does the KV cache retain?
6. When `" because"` is appended, which K/V rows are reused and which are new?

### 9.4 Parallel inside, sequential outside

Each decode iteration contains large operations that run in parallel on the
GPU. But the iterations themselves are ordered because the next selected token
is not known until the current iteration finishes.

```text
Decode iteration 1: [many parallel GPU operations] → token 1
Decode iteration 2:                               [many parallel GPU operations] → token 2
Decode iteration 3:                                                               [many parallel GPU operations] → token 3
```

This pattern combines inner parallelism with outer sequential dependency.

### 9.5 Why the shapes differ

Simplified linear-operation shapes:

```text
Prefill input: [many prompt positions, hidden size]
Decode input:  [one new position, hidden size]  # per sequence at batch 1
```

The weights are large in both cases. Prefill can reuse them across many prompt
positions in one operation. Batch-one decode has less token-position work per
iteration and repeatedly needs weights as it produces tokens one at a time.

This helps motivate—but does not universally prove—the common observation:

- Prefill often uses compute resources more effectively.
- Small-batch decode is often sensitive to memory bandwidth and launch latency.

Profiler evidence is required for a specific model and system.

The visual contrast is:

```text
PREFILL LINEAR OPERATION                    BATCH-1 DECODE LINEAR OPERATION

many token rows                             one newest-token row
┌──────────────────────┐                    ┌──────────────────────┐
│ t0: 4096 features    │                    │ t128: 4096 features  │
│ t1: 4096 features    │                    └──────────────────────┘
│ ...                  │                              ×
│ t127: 4096 features  │                    same large weight matrix
└──────────────────────┘                              ↓
          ×                                  one output-feature row
same large weight matrix
          ↓
128 output-feature rows

More rows let one operation reuse the same weights across more token positions.
```

### 9.6 Batching changes decode parallelism

If multiple sequences decode together, one iteration can process one new token
position for each sequence:

```text
Batch 1: [sequence A newest token]

Batch 4: [sequence A newest token]
         [sequence B newest token]
         [sequence C newest token]
         [sequence D newest token]
```

This gives the GPU more work per weight load and can improve aggregate
throughput, while queueing and larger batches can affect per-request latency.

In matrix form, batch size changes the number of input rows:

```text
Batch 1 decode:  X [1 sequence, 4096 features] × W [4096, 4096]
                 → Y [1 sequence, 4096 output features]

Batch 4 decode:  X [4 sequences, 4096 features] × W [4096, 4096]
                 → Y [4 sequences, 4096 output features]

Each X row belongs to a different sequence's newest token. The sequences do not
share attention histories; batching merely packages compatible work so kernels
can process more rows together.
```

### Section 9 checkpoint

Explain the phrase “decode is parallel inside each iteration but sequential
across generated tokens.” Then explain why prefill can process known prompt
positions in large tensor operations without violating causal attention.

---

## 10. When GPU Acceleration Does Not Improve the Whole Workload

### 10.1 End-to-end time includes more than arithmetic

A GPU path may include:

1. CPU preparation
2. Device-memory allocation
3. Host-to-device transfer
4. Kernel submission
5. GPU queueing and execution
6. Synchronization
7. Device-to-host transfer
8. CPU post-processing

```mermaid
flowchart LR
    PREP[CPU preparation] --> H2D[Transfer to GPU]
    H2D --> LAUNCH[Launch]
    LAUNCH --> EXEC[GPU execution]
    EXEC --> D2H[Transfer result]
    D2H --> POST[CPU post-processing]
```

Optimizing only GPU arithmetic helps only the portion of time it occupies.

### 10.2 Complete numerical example

Suppose a tiny calculation takes 20 microseconds on the CPU.

Possible GPU path:

```text
Prepare and copy input to device:       100 μs
Submit GPU kernel:                       15 μs
Wait and execute GPU arithmetic:         10 μs
Copy required result to host:           100 μs
Total:                                  225 μs
```

Although the GPU arithmetic took only 10 μs, the complete GPU path took 225 μs,
which is slower than the 20 μs CPU path.

Speedup:

```text
speedup = old time / new time
        = 20 μs / 225 μs
        ≈ 0.089×
```

A value below 1 means a slowdown.

### 10.3 Amortizing fixed overhead

If the same transfer and launch overhead applies to a much larger calculation,
useful GPU work can dominate:

```text
CPU calculation:                     50,000 μs

GPU path:
transfer + launch overhead:             215 μs
GPU execution:                         2,000 μs
total:                                 2,215 μs

speedup ≈ 50,000 / 2,215 ≈ 22.6×
```

The GPU is valuable because the workload is large enough to amortize overhead
and exploit parallelism.

### 10.4 Common reasons a GPU may not help

#### Too little work

Most execution resources remain unused, while launch overhead stays.

#### Strong sequential dependencies

Later work cannot begin until earlier results exist.

#### Frequent transfers

Moving data between host and device can dominate calculation.

#### Irregular branching

Groups of GPU threads following different paths may execute inefficiently.

#### Inefficient or missing kernels

Hardware capability matters only when software supplies an implementation that
uses it effectively.

#### Insufficient memory capacity

If required data does not fit, the system may fail or move data repeatedly
through slower paths.

#### Synchronization and CPU gaps

The GPU may sit idle while the CPU prepares work or waits unnecessarily.

### 10.5 Correctness and cost also matter

The fastest option may not be best if it:

- Produces unacceptable numerical error
- Requires expensive hardware for a small gain
- Complicates deployment or reliability
- Uses excessive energy
- Violates latency objectives while maximizing throughput

Performance engineering optimizes an objective under constraints. It does not
seek speed without context.

### Section 10 checkpoint

For any proposed GPU acceleration, list the complete end-to-end stages and
identify which costs are fixed, which grow with input size, and which could
overlap.

---

## 11. Common Misconceptions and Corrections

### “An instruction is one line of source code.”

One source statement can require many processor instructions and runtime calls.
Conversely, optimization can combine or eliminate conceptual source operations.

### “Independent operations always execute simultaneously.”

Independence permits concurrent scheduling. Physical execution depends on
available resources, the scheduler, and the implementation.

### “The order written in source code is always the exact execution order.”

Dependencies and observable behavior must be preserved, but processors,
compilers, and runtimes may reorder independent work.

### “Parallel means there is no ordering.”

Parallel algorithms contain dependency boundaries. Matrix outputs may be
independent while each output still requires products and accumulation.

### “A GPU has more cores, so it is always faster.”

CPU and GPU “core” labels are not directly comparable. Performance depends on
workload shape, implementation, data movement, precision, and the measured
boundary.

### “A GPU thread is a physical core.”

A thread is a logical instance of a GPU program. Many logical threads are
scheduled over finite physical execution resources.

### “The GPU runs the entire Python program.”

Ordinary Python runs on the CPU. Frameworks and CUDA libraries submit selected
operations to the GPU.

### “High GPU utilization proves good performance.”

A high-level activity sample does not reveal which resources are busy, whether
work is efficient, or whether the application meets its latency and throughput
goals.

### “High throughput means every request has low latency.”

Batching and queueing can improve aggregate throughput while increasing the
time experienced by an individual request.

### “If a model fits in VRAM, it should run quickly.”

Fit is a capacity question. Runtime also depends on bandwidth, compute,
scheduling, kernels, and request shape.

---

## Chapter Summary

1. Human-readable source code is translated or interpreted into lower-level
   operations.
2. Instructions operate on encoded data; arithmetic requires input values to be
   available near execution.
3. A simplified operation often follows load → compute → store.
4. Dependencies determine required logical order.
5. Independence makes parallel scheduling possible but does not guarantee
   simultaneous execution.
6. CPUs prioritize low-latency progress for a modest number of diverse,
   dependency-heavy instruction streams.
7. GPUs prioritize aggregate throughput for a large supply of structured
   parallel work.
8. Matrix multiplication contains parallel work across outputs and reductions
   within each output.
9. Transformers repeatedly apply large tensor operations, giving GPUs useful
   parallel numerical work.
10. Prefill and decode combine parallel GPU kernels with different tensor shapes
    and sequential dependencies.
11. End-to-end GPU speed depends on preparation, data movement, launch,
    execution, synchronization, and post-processing—not arithmetic alone.

---

## Vocabulary

| Term | Meaning |
| --- | --- |
| Address | A number identifying a memory location |
| Arithmetic intensity | Arithmetic work performed relative to bytes moved; developed in Lesson 05 |
| Bandwidth | Data transferred per unit time |
| Branch | An instruction that can select a different next path |
| Cache | Smaller, faster storage retaining likely-needed data near execution |
| Capacity | Amount of data or work that can be held |
| Concurrency | Multiple tasks in progress over overlapping time periods |
| Data | Encoded information read or modified by instructions |
| Data parallelism | Applying similar operations to different data items |
| Dependency | A relationship requiring one operation or result before another |
| Device | The GPU in CUDA terminology |
| Dtype | The representation and interpretation of each tensor value |
| Host | The CPU-side system in CUDA terminology |
| Index | A position used to select an element in a collection |
| Inference | Using a trained model to produce outputs for new inputs |
| Instruction | An encoded operation defined by a processor architecture |
| Latency | Elapsed time for one defined unit of work |
| Matrix | A rectangular two-dimensional collection of values |
| Parallelism | Multiple operations physically executing at the same time |
| Processor | Hardware that executes instructions |
| Program | Organized operations, data relationships, control flow, and repetition |
| Register | Very small, fast processor-local storage |
| Reduction | Combining multiple values into fewer values, such as a sum |
| Source code | Human-readable program representation written in a programming language |
| Tensor | A multidimensional collection of values with shape and dtype |
| Throughput | Completed work per unit time |
| Vector | An ordered one-dimensional collection of values |

---

## Knowledge Check

Answer in your own words without copying chapter sentences.

### Programs and execution

1. Why is `total = left + right` not itself a literal processor instruction?
2. Describe the conceptual load → add → store sequence.
3. What does an instruction decoder need to determine?
4. What is the difference between source code, a program, an instruction, and
   data?
5. Why did the chapter warn that a compact-array memory example is not a
   literal description of a Python list?

### Elementwise list addition

6. List the complete conceptual steps for calculating `C[2] = A[2] + B[2]`.
7. Which steps inside that calculation must be ordered?
8. Why is calculating `C[2]` independent of calculating `C[0]`?
9. Give four valid schedules for calculating four independent output elements.
10. Why does independence not guarantee simultaneous execution?
11. Modify the example so each output depends on the previous output.

### Processor and memory

12. What separate responsibilities belong to processor, registers, cache, and
    main memory?
13. Why can a fast arithmetic unit remain idle?
14. Explain capacity, bandwidth, and latency with separate examples.
15. What does latency hiding accomplish, and what does it not accomplish?

### Performance terms

16. Define the boundaries and units for one latency metric and one throughput
    metric relevant to inference.
17. Give an example of concurrency without physical parallel execution.
18. Construct a case where throughput improves while request latency worsens.
19. Why is batch capacity not the same as throughput?

### Dependencies and parallelism

20. Draw a dependency graph containing two independent operations followed by
    one operation that requires both results.
21. Why does a tree reduction expose more parallel work than a simple running
    sum?
22. Explain why independent work may execute in waves.
23. What limits total speedup when a significant part of the workload remains
    sequential?

### CPU and GPU design

24. What problems do branch prediction and out-of-order execution address?
25. State the broad CPU/GPU design tradeoff without saying that either processor
    is universally better.
26. Why is a GPU thread not equivalent to a physical CPU core?
27. Why does a GPU benefit from many ready thread groups?
28. List five CPU responsibilities in an inference service.

### Matrices and transformers

29. Multiply the matrices below and show every product and sum:

    ```text
    A = ┌ 2  1 ┐     B = ┌ 3  4 ┐
        └ 0  5 ┘         └ 6  2 ┘
    ```

30. Identify dependencies within one matrix output and independence across
    outputs.
31. For shapes `(M × K) × (K × N)`, what is the output shape and why must K
    match?
32. Explain how text becomes numerical tensor work.
33. Name three transformer operation categories and describe their broad work.

### Prefill, decode, and whole-system reasoning

34. Why can prompt positions be processed in large tensor operations during
    causal prefill?
35. Why must generated decode tokens be produced in sequence?
36. Explain “decode is parallel inside each iteration but sequential across
    generated tokens.”
37. Why can prefill and batch-one decode have different bottlenecks?
38. List every major stage in a complete CPU-to-GPU-to-CPU operation.
39. Using the chapter's 20 μs CPU and 225 μs GPU example, explain why faster GPU
    arithmetic did not produce faster end-to-end execution.
40. What evidence would you need before claiming a workload should move from CPU
    to GPU?

---

## Drawing Exercises

Without looking at the chapter, draw:

1. The source code → software translation → processor instruction path.
2. The load → arithmetic → store path through memory, registers, and an
   arithmetic unit.
3. The dependency graph for four independent list outputs.
4. A dependency graph with parallel branches followed by a join.
5. The host/device inference relationship.
6. The prefill/decode relationship, showing both inner parallelism and outer
   sequential dependency.

Compare your diagrams with the chapter and correct missing arrows. The arrows
are the lesson: they state which component supplies data or which result must
exist first.

## Ready to Continue When

You can narrate the full list-addition example from source-level intent through
address calculation, loads, arithmetic, stores, loop control, dependency
relationships, and possible schedules. You should also be able to explain why
the same principles scale from four list elements to millions of tensor values
without claiming that all independent operations execute at the same instant.
