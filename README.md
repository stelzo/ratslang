<p align="center">
  <h3 align="center">Ratslang</h3>
  <p align="center">A compact configuration language for physical systems.</p>
  <p align="center"><a href="https://crates.io/crates/ratslang"><img src="https://img.shields.io/crates/v/ratslang.svg" alt=""></a> <a href="https://github.com/stelzo/ratslang/tree/main/tests"><img src="https://github.com/stelzo/ratslang/actions/workflows/tests.yml/badge.svg" alt=""></a>
  </p>
</p>

Ratslang is a compact **configuration language**, delivered as a [Rust library](https://crates.io/crates/ratslang).

It was born out of frustration with the lack of proper types for **time** and **length** in configuration files. When we configure physical systems like robots, even a single zero more or less can have massive consequences. Sometimes, we don't even realize there's a problem until it's too late. The famous [Loss of the Mars Climate Orbiter](https://en.wikipedia.org/wiki/Mars_Climate_Orbiter) is a prime example of this issue.

The core motivations behind Ratslang are:

* **Solving units:** The language should inherently handle units when configuring physical systems.
* **Combining configs:** It should be easier to combine existing configuration files rather than copying them.
* **Simple and extensible:** The implementation should be small, simple, and easy to extend.
* **Type-safe units:** Physical values are stored using the [`uom`](https://crates.io/crates/uom) crate with full precision - no rounding.

Let's take a look at how it works:

~~~awk
# a comment
variable = true

time = 1s
time_is_running = 1ms..2mins # ranges convert automatically

# nanometers and micrometers for precision work
precision_length = 500nm..10um

len = 1cm..1m

# _ signals a variable the interpreter is looking for.
_internal = time_is_running

my.super.long.prefix.var = 0..100 # ranges on namespaced variable "var"

# nested
my.super {

    long.prefix {
        next_var = "UTF-🎱 Strings"
    }

    something_else = -99.018
}

mat = [ [ 6, 1, 9 ],
        [ 3, 1, 8 ] ]
~~~

Currently, Ratslang doesn't support expressions like arithmetic, loops, or conditional statements. This is a deliberate choice, and it's still undecided if it will ever include such features. Some keywords are already reserved for potential future use.

---

## Variables

* **Dynamic types**
* **Mutable**
* **Copy-on-assignment**
* **C-style naming**

---

## Types

* **Boolean**: `true`, `false`
* **Integer**: Example: `42`, `-100`
* **Floating-point**: Example: `69.42`, `-3.14`
* **String**: Quotes can be omitted if the string doesn't conflict with a previously defined variable. Example: `"my string"`, `another_string_without_quotes`
* **Path**: Example: `./somewhere/relative.dat`, `/or/absolute`, `./../backneed/dotfirst.launch.py`
* **Array/Matrix**: Newlines after commas are also supported for readability. `[ <Type>, <Type>, ... ]`, `[ 42, World, [ "nested" ] ]`, `[ [ 1, 2, 3 ], [ 4, 5, 6 ] ]`
* **Time** (stored as `uom::si::f64::Time`):
    * **Hour**: `hour`, `hours` (`s` is optional). Example: `2hours`, `1.5hour`
    * **Minute**: `min`, `mins` (`s` is optional). Example: `30min`, `5mins`
    * **Second**: `s`. Example: `10s`, `0.5s`
    * **Millisecond**: `ms`. Example: `200ms`, `1ms`
    * **Microsecond**: `us`. Example: `500us`, `1.5us`
    * **Nanosecond**: `ns`. Example: `100ns`, `1ns`
* **Length** (stored as `uom::si::f64::Length`):
    * **Meter**: `m`. Example: `10m`, `0.5m`
    * **Centimeter**: `cm`. Example: `50cm`, `2.5cm`
    * **Millimeter**: `mm`. Example: `100mm`, `1mm`
    * **Micrometer**: `um`. Example: `500um`, `1.5um`
    * **Nanometer**: `nm`. Example: `100nm`, `1nm`
* **Range**: Including unbound variants and empty `..`.
    * **Time**: Example: `1ms..5.3hours`, `6s..`
    * **Length**: Example: `1mm..100m`, `..4m`
    * **Numbers**: Example: `-4..l`, `6.00001..6.0001`

---

## Includes

In Ratslang, including is done by assigning a path to the current namespace. All variables will then get the respective prefix.

~~~awk
= ./path_relative_to_current_file.rl

strangefile {
  = ./../namespacing_contents.rl
}
~~~

---

## Library Usage

Add this to your `Cargo.toml`.

~~~toml
[dependencies]
ratslang = "0.2.0"
~~~

First, you compile a Ratslang file to get a cleaned Abstract Syntax Tree (AST) with all variables resolved.

~~~rust
let file = std::path::Path::new("./your_file.rl");
let ast = ratslang::compile_file(&file.to_path_buf(), None, None).unwrap();
~~~

Then, you can safely read the variables you need — either with the provided helper macros for concise code, or manually using Rust's powerful pattern matching.

Using helper macros (recommended):

~~~rust
use ratslang::{
    compile_file,
    resolve_string, resolve_bool, resolve_int, resolve_float,
    resolve_int_range, resolve_length_range_meters_float, resolve_time_range_seconds_float,
    resolve_path,
};

// Local configs combining user vars and optional defaults
struct Configs {
    user: ratslang::VariableHistory,
    defaults: ratslang::VariableHistory,
}

let file = std::path::Path::new("./your_file.rl");
let ast = compile_file(&file.to_path_buf(), None, None).unwrap();
let configs = Configs { user: ast.vars, defaults: ratslang::VariableHistory::new(vec![]) };

// Simple values
let name: String = resolve_string!(configs, "name")?;
let enabled: bool = resolve_bool!(configs, "variable")?;
let k: i64 = resolve_int!(configs, "k_neighbors")?;
let ratio: f64 = resolve_float!(configs, "something_else")?;

// Paths
let path: String = resolve_path!(configs, "_file")?; // e.g., `_file = /abs/or/relative`

// Ranges (with sensible defaults used when bounds are missing)
let (d_min, d_max) = resolve_length_range_meters_float!(configs, "len", 0.0, 10.0)?; // meters as f64
let (t_min, t_max) = resolve_time_range_seconds_float!(configs, "time_is_running", 0.0, 60.0)?; // seconds as f64
let (i_min, i_max) = resolve_int_range!(configs, "my.super.long.prefix.var", 0, 100)?;
~~~

Working with uom types directly:

Physical values in Ratslang are stored as `uom` types (`Length` and `Time`), giving you full access to type-safe unit conversions:

~~~rust
use ratslang::{compile_code, Rhs, Val, UnitVal, Unit};
use ratslang::{meter, millimeter, micrometer, nanometer};  // Length units
use ratslang::{second, millisecond, microsecond, nanosecond};  // Time units

let code = r#"
    sensor_range = 500um
    timeout = 1500ns
"#;

let ast = compile_code(code).unwrap();

// Get a length value and convert to any unit
if let Some(Rhs::Val(Val::UnitedVal(uv))) = ast.vars.resolve("sensor_range").unwrap() {
    // Access the underlying uom::si::f64::Length
    let length = uv.as_length().unwrap();
    
    // Convert to any unit with full precision using uom getters
    println!("Range: {} micrometers", length.get::<micrometer>());  // 500.0
    println!("Range: {} millimeters", length.get::<millimeter>());  // 0.5
    println!("Range: {} meters", length.get::<meter>());            // 0.0005
    println!("Range: {} nanometers", length.get::<nanometer>());    // 500000.0
}

// Get a time value
if let Some(Rhs::Val(Val::UnitedVal(uv))) = ast.vars.resolve("timeout").unwrap() {
    let time = uv.as_time().unwrap();
    
    // Convert to any unit with full precision using uom getters
    println!("Timeout: {} nanoseconds", time.get::<nanosecond>());   // 1500.0
    println!("Timeout: {} microseconds", time.get::<microsecond>()); // 1.5
    println!("Timeout: {} milliseconds", time.get::<millisecond>()); // 0.0015
}
~~~

Manual resolution:

~~~rust
use anyhow::anyhow;
use ratslang::{compile_file, Rhs, Val, NumVal};

// Local configs combining user vars and optional defaults
struct Configs {
    user: ratslang::VariableHistory,
    defaults: ratslang::VariableHistory,
}

let file = std::path::Path::new("./your_file.rl");
let ast = compile_file(&file.to_path_buf(), None, None).unwrap();
let vars = ast.vars.filter_ns(&["_my_namespace"]);

// Resolve a variable and pattern-match its type manually
let value = vars
    .resolve("_some_var")?
    .map_or(Ok("a_default_value".to_owned()), |rhs| {
        Ok(match rhs {
            Rhs::Val(Val::StringVal(s)) => s,
            _ => {
                return Err(anyhow!(
                    "Unexpected type for _my_namespace._some_var, expected String."
                ));
            }
        })
    })?;

// Or use the generic resolve_var! macro
use ratslang::resolve_var;
let configs = Configs { user: vars, defaults: ratslang::VariableHistory::new(vec![]) };
let k_neighbors: usize = resolve_var!(configs, k_neighborhood, as usize,
    Rhs::Val(Val::NumVal(NumVal::Integer(i))) => { i }
)?;
~~~

---

* Ratslang files typically use the `.rl` extension.
* Syntax highlighting is available with [this tree-sitter grammar](https://github.com/stelzo/tree-sitter-ratslang) or this [VS Code extension](https://marketplace.visualstudio.com/items?itemName=stelzo.ratslang). For Markdown files, you can use the `awk` language for syntax highlighting. It is not perfect but works reasonably well.
* Compile errors are beautifully rendered thanks to [Ariadne](https://crates.io/crates/ariadne) ❤️.

*Ratslang: More a slang, less a lang.*

---

## Future Plans

The following features and improvements are planned:

* **Expanded Units and Scales**: Add more unit types (mass, angle, temperature, etc.) powered by the `uom` crate.
* **Opt-in Language Versioning**: Implement an opt-in versioning system for `.rl` files.

---

## License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>
