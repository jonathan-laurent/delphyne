# MkDocs Tricks {: #main-header}

Use `!!!` to introduce admonitions and `???` for collapsible admonitions (`???+` for collapsible but expanded by default). Supported kinds include: `note`, `abstract`, `info`, `tip`, `success`, `question`, `warning`, `failure`, `danger`, `bug`, `example`, `quote`.

!!! example "You need to learn stuff"

    Here is an example.

??? note "You can expand this"

    ```py
    def fun(x):
        return x + 1
    ```

I can display some code:

```python hl_lines="2" linenums="1"
@strategy(dfs)
def my_strategy(spec): # (1)!
    prog = yield from branch(AskProg(spec))
```

1. I am not using any return type here.  <!-- Content of the annotation in the block above -->

I can show some math: \\[ \int_0^1 F \varphi(u) du. \\]

I can display keyboard keys: ++ctrl+shift+del++ or ++ctrl+period++ or ++cmd+period++.

I can also include some snippets:

```py linenums="1"
--8<-- "src/delphyne/core/chats.py"
```

Content tabs:

=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```