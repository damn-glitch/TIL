#!/bin/bash
# TIL Language Installer
# Author: Alisher Beisembekov
# Usage: curl -fsSL https://til-dev.vercel.app/install.sh | sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════╗"
echo "║              TIL INSTALLER                 ║"
echo "║     Author: Alisher Beisembekov            ║"
echo "║  \"Проще Python. Быстрее C. Умнее всех.\"    ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${BLUE}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check C compiler
echo -e "${BLUE}Checking C compiler...${NC}"
if command -v gcc &> /dev/null; then
    echo -e "${GREEN}✓ GCC found${NC}"
elif command -v clang &> /dev/null; then
    echo -e "${GREEN}✓ Clang found${NC}"
else
    echo -e "${YELLOW}⚠ No C compiler found. Install gcc or clang for compilation.${NC}"
fi

# Create directories
TIL_HOME="$HOME/.til"
TIL_BIN="$TIL_HOME/bin"
echo -e "${BLUE}Installing to ${TIL_HOME}...${NC}"
mkdir -p "$TIL_BIN"

# Download compiler
echo -e "${BLUE}Downloading TIL compiler...${NC}"
if command -v curl &> /dev/null; then
    curl -fsSL https://raw.githubusercontent.com/damn-glitch/TIL/main/src/til.py -o "$TIL_BIN/til.py"
elif command -v wget &> /dev/null; then
    wget -q https://raw.githubusercontent.com/damn-glitch/TIL/main/src/til.py -O "$TIL_BIN/til.py"
else
    echo -e "${RED}✗ Neither curl nor wget found${NC}"
    exit 1
fi

# Create wrapper script
cat > "$TIL_BIN/til" << 'EOF'
#!/bin/bash
python3 "$(dirname "$0")/til.py" "$@"
EOF
chmod +x "$TIL_BIN/til"
chmod +x "$TIL_BIN/til.py"

# Add to PATH
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    bash)
        RC_FILE="$HOME/.bashrc"
        ;;
    zsh)
        RC_FILE="$HOME/.zshrc"
        ;;
    fish)
        RC_FILE="$HOME/.config/fish/config.fish"
        mkdir -p "$(dirname "$RC_FILE")"
        ;;
    *)
        RC_FILE="$HOME/.profile"
        ;;
esac

if ! grep -q '.til/bin' "$RC_FILE" 2>/dev/null; then
    echo "" >> "$RC_FILE"
    echo '# TIL Language' >> "$RC_FILE"
    echo 'export PATH="$HOME/.til/bin:$PATH"' >> "$RC_FILE"
    echo -e "${GREEN}✓ Added to PATH in ${RC_FILE}${NC}"
fi

# Verify
export PATH="$TIL_BIN:$PATH"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         TIL INSTALLED SUCCESSFULLY!        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Version: ${CYAN}$(til --version 2>/dev/null || echo '2.0.0')${NC}"
echo ""
echo -e "${YELLOW}To start using TIL:${NC}"
echo "  1. Restart your terminal or run: source $RC_FILE"
echo "  2. Create hello.til:"
echo "     echo 'main()"
echo "         print(\"Hello, TIL!\")' > hello.til"
echo "  3. Run it: til run hello.til"
echo ""
echo -e "${CYAN}Happy coding with TIL!${NC}"
echo -e "${CYAN}Author: Alisher Beisembekov${NC}"
