//////
/// YAML Utilities
//////

import {
  parseDocument,
  Scalar,
  YAMLMap,
  YAMLSeq,
  LineCounter,
  isAlias,
} from "yaml";
import * as yaml from "yaml";
import * as vscode from "vscode";

// Convert between YAML ranges and VSCode ranges
function toVscodeRange(
  yamlRange: yaml.Range,
  lineCounter: LineCounter,
): vscode.Range {
  // The difference between `nodeEnd` and `valueEnd` is that the later includes
  // trailing comments. We do not include those comments in our ranges.
  const [start, valueEnd, nodeEnd] = yamlRange;
  const startPos = lineCounter.linePos(start);
  const endPos = lineCounter.linePos(valueEnd);
  // LineCounter uses 1-based indexing and vscode uses 0-based indexing
  return new vscode.Range(
    new vscode.Position(startPos.line - 1, startPos.col - 1),
    new vscode.Position(endPos.line - 1, endPos.col - 1),
  );
}

// Parses a YAML string and returns a result annotated wit location information. More
// precisely, for each field `x` of an object, there should be an added field `__loc__x`
// with position information. In addition, for every field `xs` that is a YAML sequence,
// we should add a `__loc_items__xs` field which is a list of the location of all items.
export function parseYamlWithLocInfo(yamlString: string): unknown {
  const lineCounter = new LineCounter();
  const doc = parseDocument(yamlString, { lineCounter });
  const anchors = new Map<string, any>();

  function addLocationInfo(node: any, path: string[] = []): any {
    if (isAlias(node)) {
      const anchorNode = anchors.get(node.source);
      if (anchorNode) {
        return addLocationInfo(anchorNode.node, path);
      } else {
        return null;
      }
    }

    if (node.anchor) {
      anchors.set(node.anchor, { node, path });
    }

    if (node instanceof YAMLMap) {
      const obj: any = {};
      obj["__loc"] = node.range ? toVscodeRange(node.range, lineCounter) : null;
      for (const item of node.items) {
        const key = item.key.value;
        const value = item.value;
        const newPath = [...path, key];
        obj[key] = addLocationInfo(value, newPath);

        if (value?.range) {
          obj[`__loc__${key}`] = toVscodeRange(value.range, lineCounter);
        }

        // Check if the value is a sequence to handle `__loc_items__`
        if (value instanceof YAMLSeq) {
          obj[`__loc_items__${key}`] = value.items.map((item) =>
            item?.range ? toVscodeRange(item.range, lineCounter) : null,
          );
        }
      }
      return obj;
    } else if (node instanceof YAMLSeq) {
      return node.items.map((item, index) =>
        addLocationInfo(item, [...path, index.toString()]),
      );
    } else if (node instanceof Scalar) {
      return node.value;
    } else {
      return node;
    }
  }
  return addLocationInfo(doc.contents);
}

// Serializes an object obtained with `parseYamlWithLocInfo`, dumping location info.
export function serializeWithoutLocInfo(obj: any, indent: number = 0): string {
  function customReplacer(key: string, value: any): any {
    // If the key starts with '__loc', ignore this key-value pair
    if (key.startsWith("__loc")) {
      return undefined;
    }
    return value;
  }

  return JSON.stringify(obj, customReplacer, indent);
}

//////
/// YAML Pretty Printing
//////

const MAX_FLOW_SEQ_LENGTH = 50;

const setFlowForSmallSequences = (_key: any, node: any) => {
  if (yaml.isSeq(node)) {
    const allScalars = node.items.every((item) => yaml.isScalar(item));
    if (allScalars) {
      const rendered = new yaml.Document(node).toString({
        collectionStyle: "flow",
        flowCollectionPadding: false,
      });
      if (rendered.length <= MAX_FLOW_SEQ_LENGTH) {
        node.flow = true;
      }
    }
  }
};

export function prettyYaml(obj: any): string {
  const doc: yaml.Document = new yaml.Document(obj);
  yaml.visit(doc, setFlowForSmallSequences);
  return doc.toString({ flowCollectionPadding: false, lineWidth: 0 });
}

export function prettyYamlOneLiner(obj: any): string {
  // Dump into YAML, using no new lines
  const doc: yaml.Document = new yaml.Document(obj);
  return doc
    .toString({
      blockQuote: false,
      collectionStyle: "flow",
      flowCollectionPadding: false,
      lineWidth: 0,
    })
    .trim();
}
