#!/usr/bin/env sh

echo -e "# List of available commands\n\`\`\`" > commands.md
poetry run methscan --help >> commands.md
echo -e "\`\`\`\n# prepare\n\`\`\`" >> commands.md
poetry run methscan prepare --help >> commands.md

echo -e "\`\`\`\n# filter\n\`\`\`" >> commands.md
poetry run methscan filter --help >> commands.md

echo -e "\`\`\`\n# smooth\n\`\`\`" >> commands.md
poetry run methscan smooth --help >> commands.md

echo -e "\`\`\`\n# scan\n\`\`\`" >> commands.md
poetry run methscan scan --help >> commands.md

echo -e "\`\`\`\n# diff\n\`\`\`" >> commands.md
poetry run methscan diff --help >> commands.md

echo -e "\`\`\`\n# matrix\n\`\`\`" >> commands.md
poetry run methscan matrix --help >> commands.md

echo -e "\`\`\`\n# profile\n\`\`\`" >> commands.md
poetry run methscan profile --help >> commands.md
echo -e "\`\`\`" >> commands.md
