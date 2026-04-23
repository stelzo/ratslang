# Ratslang LSP

Language Server Protocol implementation for [Ratslang](https://codeberg.org/stelzo/ratslang).

## Building

```bash
cargo build --release
```

The binary will be at `target/release/ratslang-lsp`.

## Editor Configuration

### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "ratslang"
scope = "source.rl"
file-types = ["rl"]
roots = []
comment-token = "#"

[language.language-server]
command = "/path/to/ratslang-lsp"
```

### Neovim

#### Using nvim-lspconfig

```lua
local lspconfig = require('lspconfig')

lspconfig.ratslang = {
  default_config = {
    cmd = { '/path/to/ratslang-lsp' },
    filetypes = { 'rl' },
    root_dir = lspconfig.util.find_git_ancestor,
    single_file_support = true,
  },
}

lspconfig.ratslang.setup({})
```

#### Using Lazy.nvim

```lua
{
  'neovim/nvim-lspconfig',
  config = function()
    local lspconfig = require('lspconfig')
    
    lspconfig.ratslang = {
      default_config = {
        cmd = { '/path/to/ratslang-lsp' },
        filetypes = { 'rl' },
        root_dir = lspconfig.util.find_git_ancestor,
        single_file_support = true,
      },
    }
    
    lspconfig.ratslang.setup({})
  end,
}
```

Replace `/path/to/ratslang-lsp` with the actual path to your built binary.

## Features

- Completions
- Hover documentation
- Go to definition
- Document symbols
- Code actions
