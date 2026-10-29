"""Hand-written buggy / fixed code pairs.

Used for three things:
  1. the sample gallery in the UI
  2. a small cross-language probe set (model trained on C, tested on Python and C)
  3. an offline reference answer when the Gemini API is unreachable

`lines` are 1-indexed line numbers of the bug inside `buggy`.
"""

SAMPLES = [
    # ------------------------------------------------------------------ Python
    {
        "id": "py-binary-search",
        "title": "Binary search that misses the last element",
        "language": "python",
        "bug_type": "Off-by-one error",
        "severity": "Major",
        "lines": [3],
        "explanation": "The loop stops when lo == hi, so the last remaining candidate is never checked. Searching for the only element left in range returns -1.",
        "buggy": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
        "fixed": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
    },
    {
        "id": "py-mutable-default",
        "title": "Shared list across calls",
        "language": "python",
        "bug_type": "Mutable default argument",
        "severity": "Major",
        "lines": [1],
        "explanation": "The default list is created once when the function is defined, so every call without a bucket keeps appending to the same list.",
        "buggy": """def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket
""",
        "fixed": """def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
""",
    },
    {
        "id": "py-average",
        "title": "Average of an empty list",
        "language": "python",
        "bug_type": "Division by zero",
        "severity": "Major",
        "lines": [2],
        "explanation": "When nums is empty, len(nums) is 0 and the function crashes with ZeroDivisionError.",
        "buggy": """def average(nums):
    return sum(nums) / len(nums)
""",
        "fixed": """def average(nums):
    if not nums:
        return 0.0
    return sum(nums) / len(nums)
""",
    },
    {
        "id": "py-remove-while-iterating",
        "title": "Removing items while looping",
        "language": "python",
        "bug_type": "Collection modified during iteration",
        "severity": "Major",
        "lines": [2, 4],
        "explanation": "Removing from a list while iterating over it shifts the remaining items, so the element right after each removed one is skipped. [-1, -2, 3] returns [-2, 3].",
        "buggy": """def remove_negatives(nums):
    for n in nums:
        if n < 0:
            nums.remove(n)
    return nums
""",
        "fixed": """def remove_negatives(nums):
    return [n for n in nums if n >= 0]
""",
    },
    {
        "id": "py-is-compare",
        "title": "Comparing strings with is",
        "language": "python",
        "bug_type": "Identity vs equality",
        "severity": "Major",
        "lines": [2],
        "explanation": "`is` checks whether two objects are the same object in memory, not whether they are equal. A role string read from a file or request can equal 'admin' and still fail this check.",
        "buggy": """def is_admin(role):
    return role is "admin"
""",
        "fixed": """def is_admin(role):
    return role == "admin"
""",
    },
    {
        "id": "py-file-leak",
        "title": "File left open on early return",
        "language": "python",
        "bug_type": "Resource leak",
        "severity": "Minor",
        "lines": [2, 5],
        "explanation": "If the file is empty the function returns before f.close(), leaking the file handle. Using a with block closes it on every path.",
        "buggy": """def read_config(path):
    f = open(path)
    data = f.read()
    if not data:
        return {}
    f.close()
    return json.loads(data)
""",
        "fixed": """def read_config(path):
    with open(path) as f:
        data = f.read()
    if not data:
        return {}
    return json.loads(data)
""",
    },
    {
        "id": "py-sql-injection",
        "title": "User lookup built with an f-string",
        "language": "python",
        "bug_type": "SQL injection",
        "severity": "Critical",
        "lines": [2],
        "explanation": "The username is pasted straight into the SQL text. Input like ' OR '1'='1 changes the query and can dump or modify the whole table.",
        "buggy": """def get_user(cursor, username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
""",
        "fixed": """def get_user(cursor, username):
    query = "SELECT * FROM users WHERE name = %s"
    cursor.execute(query, (username,))
    return cursor.fetchone()
""",
    },
    {
        "id": "py-pairwise",
        "title": "Pairwise sums run past the end",
        "language": "python",
        "bug_type": "Index out of range",
        "severity": "Major",
        "lines": [3, 4],
        "explanation": "On the last iteration i + 1 equals len(nums), so nums[i + 1] raises IndexError.",
        "buggy": """def pairwise_sums(nums):
    result = []
    for i in range(len(nums)):
        result.append(nums[i] + nums[i + 1])
    return result
""",
        "fixed": """def pairwise_sums(nums):
    result = []
    for i in range(len(nums) - 1):
        result.append(nums[i] + nums[i + 1])
    return result
""",
    },
    {
        "id": "py-floor-division",
        "title": "Temperature conversion loses precision",
        "language": "python",
        "bug_type": "Wrong operator",
        "severity": "Minor",
        "lines": [2],
        "explanation": "// is floor division, so 37.5 C becomes 99.0 F instead of 99.5 F.",
        "buggy": """def to_fahrenheit(celsius):
    return celsius * 9 // 5 + 32
""",
        "fixed": """def to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32
""",
    },
    {
        "id": "py-missing-return",
        "title": "Recursive factorial returns None",
        "language": "python",
        "bug_type": "Missing return",
        "severity": "Major",
        "lines": [4],
        "explanation": "The recursive result is computed but never returned, so factorial(3) returns None.",
        "buggy": """def factorial(n):
    if n == 0:
        return 1
    n * factorial(n - 1)
""",
        "fixed": """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
""",
    },
    {
        "id": "py-bare-except",
        "title": "Silent failure on bad input",
        "language": "python",
        "bug_type": "Swallowed exception",
        "severity": "Minor",
        "lines": [4, 5],
        "explanation": "A bare except hides every error, even Ctrl+C, and the function silently returns None for bad input.",
        "buggy": """def safe_int(value):
    try:
        return int(value)
    except:
        pass
""",
        "fixed": """def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
""",
    },
    {
        "id": "py-find-max",
        "title": "Max of negative numbers",
        "language": "python",
        "bug_type": "Wrong initial value",
        "severity": "Major",
        "lines": [2],
        "explanation": "Starting from 0 means a list of only negative numbers returns 0, which is not in the list.",
        "buggy": """def find_max(nums):
    best = 0
    for n in nums:
        if n > best:
            best = n
    return best
""",
        "fixed": """def find_max(nums):
    best = nums[0]
    for n in nums[1:]:
        if n > best:
            best = n
    return best
""",
    },
    {
        "id": "py-grid-alias",
        "title": "Grid rows that change together",
        "language": "python",
        "bug_type": "Aliased references",
        "severity": "Major",
        "lines": [2],
        "explanation": "Multiplying the outer list repeats the same row object, so setting grid[0][0] changes every row.",
        "buggy": """def make_grid(rows, cols):
    return [[0] * cols] * rows
""",
        "fixed": """def make_grid(rows, cols):
    return [[0] * cols for _ in range(rows)]
""",
    },
    # ------------------------------------------------------------------ C
    {
        "id": "c-strcpy-overflow",
        "title": "Name copied into a fixed buffer",
        "language": "c",
        "bug_type": "Buffer overflow",
        "severity": "Critical",
        "lines": [2, 3],
        "explanation": "strcpy copies until the null byte with no size limit. A name longer than 15 characters writes past buf and corrupts the stack.",
        "buggy": """void copy_name(char *dst, const char *src) {
    char buf[16];
    strcpy(buf, src);
    strcpy(dst, buf);
}
""",
        "fixed": """void copy_name(char *dst, size_t dst_size, const char *src) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%s", src);
    snprintf(dst, dst_size, "%s", buf);
}
""",
    },
    {
        "id": "c-use-after-free",
        "title": "Node read after it is freed",
        "language": "c",
        "bug_type": "Use after free",
        "severity": "Critical",
        "lines": [3, 4, 5],
        "explanation": "n is freed and then n->next is read. That memory may already be reused, which gives wrong results or a crash and is often exploitable.",
        "buggy": """int process(struct node *n) {
    int val = n->value;
    free(n);
    if (n->next != NULL)
        return val + n->next->value;
    return val;
}
""",
        "fixed": """int process(struct node *n) {
    int val = n->value;
    struct node *next = n->next;
    free(n);
    if (next != NULL)
        return val + next->value;
    return val;
}
""",
    },
    {
        "id": "c-packet-length",
        "title": "Packet length trusted from input",
        "language": "c",
        "bug_type": "Out-of-bounds read",
        "severity": "Critical",
        "lines": [2, 3],
        "explanation": "size comes from the packet itself and is never checked against len or the output size, so memcpy can read and write past both buffers.",
        "buggy": """static int read_header(const uint8_t *buf, int len, uint8_t *out) {
    int size = buf[0];
    memcpy(out, buf + 1, size);
    return size;
}
""",
        "fixed": """static int read_header(const uint8_t *buf, int len, uint8_t *out, int out_size) {
    if (len < 1)
        return -1;
    int size = buf[0];
    if (size > len - 1 || size > out_size)
        return -1;
    memcpy(out, buf + 1, size);
    return size;
}
""",
    },
    {
        "id": "c-malloc-null",
        "title": "String copy without checking malloc",
        "language": "c",
        "bug_type": "Null pointer dereference",
        "severity": "Major",
        "lines": [2, 3],
        "explanation": "malloc can return NULL when memory runs out, and strcpy then writes through a null pointer.",
        "buggy": """char *dup_string(const char *s) {
    char *copy = malloc(strlen(s) + 1);
    strcpy(copy, s);
    return copy;
}
""",
        "fixed": """char *dup_string(const char *s) {
    char *copy = malloc(strlen(s) + 1);
    if (copy == NULL)
        return NULL;
    strcpy(copy, s);
    return copy;
}
""",
    },
    {
        "id": "c-alloc-overflow",
        "title": "Table size multiplied without a check",
        "language": "c",
        "bug_type": "Integer overflow",
        "severity": "Critical",
        "lines": [2, 3, 4],
        "explanation": "count * sizeof(int) can overflow and allocate a tiny buffer, then the loop writes count items into it. A negative count is also accepted.",
        "buggy": """int *alloc_table(int count) {
    int *table = malloc(count * sizeof(int));
    for (int i = 0; i < count; i++)
        table[i] = 0;
    return table;
}
""",
        "fixed": """int *alloc_table(int count) {
    if (count <= 0 || (size_t)count > SIZE_MAX / sizeof(int))
        return NULL;
    int *table = calloc((size_t)count, sizeof(int));
    return table;
}
""",
    },
]


def normalize(code):
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def find_sample(code):
    """Return (sample, 'buggy' | 'fixed') if the code matches a known sample."""
    key = normalize(code)
    for s in SAMPLES:
        if normalize(s["buggy"]) == key:
            return s, "buggy"
        if normalize(s["fixed"]) == key:
            return s, "fixed"
    return None, None


def probe_set():
    """Flat labelled list: buggy version = 1, fixed version = 0."""
    rows = []
    for s in SAMPLES:
        rows.append({"id": s["id"] + ":buggy", "language": s["language"], "func": s["buggy"], "target": 1})
        rows.append({"id": s["id"] + ":fixed", "language": s["language"], "func": s["fixed"], "target": 0})
    return rows


if __name__ == "__main__":
    import json
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "data", "probe_set.json")
    with open(out, "w") as f:
        json.dump(probe_set(), f, indent=1)
    print("wrote", len(probe_set()), "probe rows to", os.path.normpath(out))
