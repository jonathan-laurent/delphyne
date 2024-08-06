# Generate the demo file schema
SCHEMA_FOLDER="vscode-ui/resources"
mkdir -p $SCHEMA_FOLDER
python -m delphyne.server.generate_schemas demo_file > $SCHEMA_FOLDER/demo-schema.json

# Generate stubs by using GPT-4 to translae
STUBS_FOLDER="vscode-ui/src/stubs"
python -m delphyne.server.generate_stubs demos > $STUBS_FOLDER/demos.ts
python -m delphyne.server.generate_stubs feedback > $STUBS_FOLDER/feedback.ts