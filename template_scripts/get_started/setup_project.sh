#!/bin/bash

# ==============================================================================
# Python Project Template - Quick Setup Script
# ==============================================================================
#
# COLOR OUTPUT:
#   - Colors are automatically detected and enabled for supported terminals
#   - Use --no-color to disable colors
#   - Set NO_COLOR=1 environment variable to disable colors globally
#   - Colors work on macOS, Linux, and Windows (Git Bash/WSL)
#
#
# This script helps you quickly customize your new Python project created
# from the python-project-template repository.
#
# USAGE:
#   ./template_scripts/get_started/setup_project.sh "my-awesome-project"
#   ./template_scripts/get_started/setup_project.sh "my-awesome-project" --help
#
# WHAT IT DOES:
#   1. Updates pyproject.toml with your project name
#   2. Renames the source directory to match your package name
#   3. Updates the README title
#   4. Provides next steps for development
#
# EXAMPLES:
#   ./scripts/get_started/setup_project.sh "data-analyzer"
#   ./scripts/get_started/setup_project.sh "web-scraper"
#   ./scripts/get_started/setup_project.sh "ml-pipeline"
#
# ==============================================================================

set -e  # Exit on any error

# Colors for output - with fallback for non-color terminals
if [[ -t 1 ]] && [[ -n "$TERM" ]] && command -v tput >/dev/null 2>&1; then
    # Terminal supports colors
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    BOLD=$(tput bold)
    NC=$(tput sgr0)
else
    # No color support or not a terminal
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# Check for NO_COLOR environment variable
if [[ -n "$NO_COLOR" ]]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# Debug: Print color status (only in verbose mode or when debugging)
if [[ "${DEBUG_COLORS:-false}" == "true" ]]; then
    echo "Color support: RED='$RED' GREEN='$GREEN' YELLOW='$YELLOW' BLUE='$BLUE' BOLD='$BOLD' NC='$NC'"
fi

# Function to print colored output
print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_header() {
    echo -e "${BOLD}${BLUE}$1${NC}"
}

# Function to show help
show_help() {
    cat << EOF
${BOLD}Python Project Template - Quick Setup Script${NC}

${BOLD}USAGE:${NC}
    ./template_scripts/get_started/setup_project.sh PROJECT_NAME [OPTIONS]

${BOLD}ARGUMENTS:${NC}
    PROJECT_NAME    Your project name (use kebab-case like "my-awesome-project")

${BOLD}OPTIONS:${NC}
    -h, --help      Show this help message
    --dry-run       Show what would be changed without making changes
    --verbose       Show detailed output
    --no-color      Disable colored output

${BOLD}ENVIRONMENT VARIABLES:${NC}
    NO_COLOR=1      Disable colored output globally
    DEBUG_COLORS=1  Enable color debugging (shows color codes)

${BOLD}EXAMPLES:${NC}
    ./template_scripts/get_started/setup_project.sh "data-analyzer"
    ./template_scripts/get_started/setup_project.sh "web-scraper" --verbose
    ./template_scripts/get_started/setup_project.sh "ml-pipeline" --dry-run

${BOLD}WHAT THIS SCRIPT DOES:${NC}
    1. Validates your project name
    2. Updates pyproject.toml with project details
    3. Renames src/python_project_template/ to match your package
    4. Updates README.md title
    5. Shows next steps for development

${BOLD}PROJECT NAME CONVENTIONS:${NC}
    • Use kebab-case: "my-awesome-project"
    • Will be converted to:
      - Package name: my_awesome_project (snake_case)
      - Script name: myawesomeproject (no separators)
      - Directory: src/my_awesome_project/

${BOLD}REQUIREMENTS:${NC}
    • Run from the root of your project directory
    • Ensure you have write permissions
    • Works on macOS, Linux, and Windows (Git Bash/WSL)

EOF
}

# Function to validate project name
validate_project_name() {
    local name="$1"
    
    if [[ -z "$name" ]]; then
        print_error "Project name cannot be empty"
        return 1
    fi
    
    if [[ ! "$name" =~ ^[a-z0-9-]+$ ]]; then
        print_error "Project name should only contain lowercase letters, numbers, and hyphens"
        print_info "Good: my-project, data-analyzer, web-app"
        print_info "Bad: My-Project, data_analyzer, web app"
        return 1
    fi
    
    if [[ "$name" =~ ^- ]] || [[ "$name" =~ -$ ]]; then
        print_error "Project name cannot start or end with a hyphen"
        return 1
    fi
    
    if [[ "$name" =~ -- ]]; then
        print_error "Project name cannot contain consecutive hyphens"
        return 1
    fi
    
    return 0
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found. Please run this script from the project root directory."
        return 1
    fi
    
    if [[ ! -d "src/python_project_template" ]]; then
        print_warning "src/python_project_template/ directory not found."
        print_info "This might be normal if you've already renamed it or are using a different structure."
    fi
    
    return 0
}

# Function to convert project name to different formats
convert_names() {
    local project_name="$1"
    
    # Convert to package name (snake_case)
    PACKAGE_NAME=$(echo "$project_name" | sed 's/-/_/g')
    
    # Convert to script name (no separators)
    SCRIPT_NAME=$(echo "$project_name" | sed 's/-//g')
    
    # Convert to class name (PascalCase)
    CLASS_NAME=$(echo "$project_name" | sed 's/-/ /g' | sed 's/\b\w/\U&/g' | sed 's/ //g')
}

# Function to update pyproject.toml
update_pyproject() {
    local project_name="$1"
    local package_name="$2"
    local script_name="$3"
    local dry_run="$4"
    
    if [[ "$dry_run" == "true" ]]; then
        print_info "Would update pyproject.toml:"
        print_info "  - name: '$project_name'"
        print_info "  - script: '$script_name = \"$package_name:main\"'"
        print_info "  - known-first-party: ['$package_name']"
        return 0
    fi
    
    # Create backup
    cp pyproject.toml pyproject.toml.backup
    
    # Update project name (replace the vanilla template name)
    sed -i.tmp "s/name = \"python-project-template\"/name = \"$project_name\"/g" pyproject.toml
    print_success "Updated project name to: $project_name"
    
    # Update script name (replace the vanilla template script name)
    sed -i.tmp "s/project-template = \"python_project_template:main\"/$script_name = \"$package_name:main\"/g" pyproject.toml
    print_success "Updated script name to: $script_name"
    
    # Update package name in script reference
    sed -i.tmp "s/python_project_template:main/$package_name:main/g" pyproject.toml
    print_success "Updated package reference to: $package_name"
    
    # Update known-first-party for ruff
    sed -i.tmp "s/known-first-party = \[\"python_project_template\"\]/known-first-party = [\"$package_name\"]/g" pyproject.toml
    print_success "Updated ruff known-first-party to: $package_name"
    
    # Clean up temporary files
    rm -f pyproject.toml.tmp
}

# Function to rename source directory
rename_source_directory() {
    local package_name="$1"
    local dry_run="$2"
    
    local old_dir="src/python_project_template"
    local new_dir="src/$package_name"
    
    if [[ ! -d "$old_dir" ]]; then
        print_warning "Source directory $old_dir not found - might already be renamed"
        return 0
    fi
    
    if [[ -d "$new_dir" ]]; then
        print_warning "Target directory $new_dir already exists - skipping rename"
        return 0
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        print_info "Would rename: $old_dir → $new_dir"
        return 0
    fi
    
    mv "$old_dir" "$new_dir"
    print_success "Renamed source directory: $old_dir → $new_dir"
}

# Function to update README
update_readme() {
    local project_name="$1"
    local dry_run="$2"
    
    if [[ "$dry_run" == "true" ]]; then
        print_info "Would update README.md title to: # $project_name"
        return 0
    fi
    
    # Create backup
    cp README.md README.md.backup
    
    # Update title (first line starting with #)
    sed -i.tmp "1s/^# .*/# $project_name/" README.md
    print_success "Updated README title to: # $project_name"
    
    # Clean up temporary files
    rm -f README.md.tmp
}

# Function to show next steps
show_next_steps() {
    local project_name="$1"
    local package_name="$2"
    local script_name="$3"
    
    print_header "🎉 Project Setup Complete!"
    echo
    print_info "Your project '$project_name' has been customized with:"
    echo "   📦 Package name: $package_name"
    echo "   🔧 Script name: $script_name"
    echo "   📁 Source directory: src/$package_name/"
    echo
    
    print_header "📋 Next Steps:"
    echo "   1. Review the changes made to your files"
    echo "   2. Update the project description in pyproject.toml"
    echo "   3. Customize the README.md content"
    echo "   4. Install dependencies:"
    echo "      ${BLUE}uv sync --group dev${NC}"
    echo "   5. Install in editable mode:"
    echo "      ${BLUE}uv pip install -e .${NC}"
    echo "   6. Initialize git repository:"
    echo "      ${BLUE}git init && git add . && git commit -m 'Initial commit'${NC}"
    echo "   7. Start coding in src/$package_name/"
    echo
    
    print_info "Backup files created:"
    echo "   • pyproject.toml.backup"
    echo "   • README.md.backup"
    echo
    print_info "You can remove these backup files once you're satisfied with the changes."
}

# Main function
main() {
    local project_name=""
    local dry_run="false"
    local verbose="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --verbose)
                verbose="true"
                shift
                ;;
            --no-color)
                disable_colors
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                if [[ -z "$project_name" ]]; then
                    project_name="$1"
                else
                    print_error "Too many arguments. Expected one project name."
                    echo "Use --help for usage information"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if project name is provided
    if [[ -z "$project_name" ]]; then
        print_error "Project name is required"
        echo
        echo "Usage: $0 PROJECT_NAME [OPTIONS]"
        echo "Use --help for more information"
        exit 1
    fi
    
    # Validate inputs
    if ! validate_project_name "$project_name"; then
        exit 1
    fi
    
    if ! check_directory; then
        exit 1
    fi
    
    # Convert names
    convert_names "$project_name"
    
    if [[ "$verbose" == "true" ]] || [[ "$dry_run" == "true" ]]; then
        print_info "Project name conversions:"
        echo "   Original: $project_name"
        echo "   Package:  $PACKAGE_NAME"
        echo "   Script:   $SCRIPT_NAME"
        echo "   Class:    $CLASS_NAME"
        echo
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        print_header "🔍 Dry Run - No Changes Will Be Made"
        echo
    else
        print_header "🚀 Setting up project: $project_name"
        echo
    fi
    
    # Perform updates
    update_pyproject "$project_name" "$PACKAGE_NAME" "$SCRIPT_NAME" "$dry_run"
    rename_source_directory "$PACKAGE_NAME" "$dry_run"
    update_readme "$project_name" "$dry_run"
    
    if [[ "$dry_run" == "true" ]]; then
        echo
        print_info "This was a dry run. No files were changed."
        print_info "Run without --dry-run to apply these changes."
    else
        echo
        show_next_steps "$project_name" "$PACKAGE_NAME" "$SCRIPT_NAME"
    fi
}

# Run main function with all arguments
main "$@"
