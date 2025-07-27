import * as assert from "assert";
import * as yaml from "yaml";
import { parse } from "yaml";
import {
  parseYamlWithLocInfo,
  serializeWithoutLocInfo,
  prettyYaml,
} from "../yaml_utils";

const TEST_YAML = `
name: John
age: 30
skills:
- skill: programming
  rating: 5
- skill: debugging
  rating: 4`;

suite("YAML Utils Test Suite", () => {
  test("Parse with Location Info", () => {
    const result = parseYamlWithLocInfo(TEST_YAML) as any;
    const serialized = serializeWithoutLocInfo(result, 2);
    const serialized_bis = JSON.stringify(parse(TEST_YAML), null, 2);
    assert.strictEqual(serialized, serialized_bis);
    assert.strictEqual(result.name, "John");
    assert.strictEqual(result.__loc.start.line, 1);
    assert.strictEqual(result.__loc.end.line, 7);
    assert.strictEqual(result.__loc__name.start.line, 1);
    assert.strictEqual(result.__loc__age.start.line, 2);
  });
});

suite("Test breaking out of TS generators", () => {
  test("Test breaking out of TS generators", async () => {
    function* gen() {
      try {
        yield 1;
        yield 2;
        yield 3;
      } finally {
        console.log("Finally");
      }
    }

    console.log("Hello");
    let i = 0;
    for (const x of gen()) {
      i++;
      console.log("Inside", x);
      if (i > 1) {
        break;
      }
    }
    console.log("Outside");
  });
});

const COMPLEX_YAML = `
hello: |
  Hello, this is a multiline string
  that spans multiple lines.
world: a simpler string
list: [foo, bar]
`;

suite("YAML Pretty Printer", () => {
  test("Parse example 1", () => {
    const ex = yaml.parse(COMPLEX_YAML);
    console.log(prettyYaml(ex));
    console.log(
      prettyYaml({ foo: Array.from({ length: 15 }, (_, i) => i + 1) }),
    );
  });
});

const YAML_WITH_ANCHORS = `
hello:
  foo: &foo {list: [1, 2, 3, 4]}
  bar: *foo
`;

suite("Parse YAML with anchors", () => {
  test("Parse example 1", () => {
    const ex = parseYamlWithLocInfo(YAML_WITH_ANCHORS);
    console.log(ex);
  });
});

// --- extractConfigBlock tests ---
import { extractConfigBlock } from "../config";

suite("extractConfigBlock", () => {
  test("Extracts simple config block", () => {
    const doc = `# @config\n# foo: bar\n# baz: qux\n# @end\nother: value`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, "foo: bar\nbaz: qux");
  });

  test("Handles blank lines and comments before block", () => {
    const doc = `\n#\n# @config\n# foo: bar\n# @end`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, "foo: bar");
  });

  test("Returns null if @config not at top", () => {
    const doc = `foo: bar\n# @config\n# baz: qux\n# @end`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, null);
  });

  test("Returns null if @end missing", () => {
    const doc = `# @config\n# foo: bar`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, null);
  });

  test("Handles empty config block", () => {
    const doc = `# @config\n# @end`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, "");
  });

  test("Handles lines with just #", () => {
    const doc = `# @config\n# foo: bar\n#\n# baz: qux\n# @end`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, "foo: bar\n\nbaz: qux");
  });

  test("Returns null if non-comment before @config", () => {
    const doc = `\nfoo: bar\n# @config\n# baz: qux\n# @end`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, null);
  });

  test("Returns null if non-comment inside block", () => {
    const doc = `# @config\n# foo: bar\nbaz: qux\n# @end`;
    const result = extractConfigBlock(doc);
    assert.strictEqual(result, null);
  });
});
