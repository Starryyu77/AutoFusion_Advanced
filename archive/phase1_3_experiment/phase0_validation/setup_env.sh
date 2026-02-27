#!/bin/bash
# Setup environment for Real API experiments

echo "=========================================="
echo "Setup Environment for Real API"
echo "=========================================="
echo ""

# Check if already set
if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo "DEEPSEEK_API_KEY is already set: ${DEEPSEEK_API_KEY:0:8}..."
    echo ""
    read -p "Do you want to update it? (y/n): " update
    if [ "$update" != "y" ]; then
        echo "Keeping existing API key."
        exit 0
    fi
fi

echo "Please enter your DeepSeek API Key:"
echo "(Get it from: https://platform.deepseek.com/api_keys)"
read -s api_key

if [ -z "$api_key" ]; then
    echo "ERROR: API key cannot be empty!"
    exit 1
fi

# Add to ~/.bashrc for persistence
if ! grep -q "DEEPSEEK_API_KEY" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# DeepSeek API Key for AutoFusion" >> ~/.bashrc
    echo "export DEEPSEEK_API_KEY='$api_key'" >> ~/.bashrc
    echo "export BUDGET_LIMIT_YUAN=10000" >> ~/.bashrc
    echo "export CACHE_DIR='/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/.cache/llm'" >> ~/.bashrc
    echo "Added to ~/.bashrc"
else
    # Update existing key
    sed -i "/DEEPSEEK_API_KEY/c\export DEEPSEEK_API_KEY='$api_key'" ~/.bashrc
    echo "Updated ~/.bashrc"
fi

# Export for current session
export DEEPSEEK_API_KEY="$api_key"
export BUDGET_LIMIT_YUAN=10000
export CACHE_DIR="/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/.cache/llm"

echo ""
echo "✓ API Key configured: ${api_key:0:8}..."
echo "✓ Budget limit: 10000 yuan"
echo "✓ Cache directory: $CACHE_DIR"
echo ""
echo "Note: Run 'source ~/.bashrc' to apply to current session"
echo "      Or log out and log back in for permanent effect"
echo ""
echo "Next step: Run validation experiment"
echo "  bash experiment/phase0_validation/run_validation.sh"
