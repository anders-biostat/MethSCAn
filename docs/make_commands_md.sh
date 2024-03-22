#!/usr/bin/env sh

echo -e "# List of available commands\n\`\`\`" > commands.md
methscan --help >> commands.md
echo -e "\`\`\`\n# prepare\n\`\`\`" >> commands.md
methscan prepare --help >> commands.md

echo -e "\`\`\`\n# filter\n\`\`\`" >> commands.md
methscan filter --help >> commands.md

echo -e "\`\`\`\n# smooth\n\`\`\`" >> commands.md
methscan smooth --help >> commands.md

echo -e "\`\`\`\n# scan\n\`\`\`" >> commands.md
methscan scan --help >> commands.md

echo -e "\`\`\`\n# diff\n\`\`\`" >> commands.md
methscan diff --help >> commands.md

echo -e "\`\`\`\n# matrix\n\`\`\`" >> commands.md
methscan matrix --help >> commands.md

echo -e "\`\`\`\n# profile\n\`\`\`" >> commands.md
methscan profile --help >> commands.md
echo -e "\`\`\`" >> commands.md
