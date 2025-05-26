# Ratslang

*More a slang, less a lang.*

Ratslang is a compact **configuration language**, delivered as a Rust-based parser library.

It was born out of frustration with the lack of proper types for **time** and **length** in configuration files. When we configure physical systems, even a single zero more or less can have massive consequences. Sometimes, we don't even realize there's a problem until it's too late. The famous [Loss of the Mars Climate Orbiter](https://spacemath.gsfc.nasa.gov/weekly/6Page53.pdf) is a prime example of this issue.

The core motivations behind Ratslang are:

* **Solving units:** The language should inherently handle units when configuring physical systems.
* **Combining configs:** It should be easier to combine existing configuration files rather than copying them.
* **Simple and extensible:** The implementation should be small, simple, and easy to extend.

Let's take a look at how it works:

~~~awk
# a comment
variable = true

time = 1s
time_is_running = 1ms..2mins # ranges convert automatically

length = 1cm..1m

# _ signals a variable the interpreter is looking for.
_internal = time_is_running

my.super.long.prefix.var = 0..100 # ranges on namespaced variable "var"

# nested
my.super.{
  
  long.prefix.{
    next_var = 3.
  }

  something_else = -99.018
}
~~~

Currently, Ratslang doesn't support expressions like arithmetic, loops, or conditional statements. This is a deliberate choice, and it's still undecided if it will ever include such features. Some keywords are already reserved for potential future use.

---

## Variables

* **Dynamic types**
* **Overwritable** (as you'd expect)
* **Copy-on-assignment**
* **C-style naming**

---

## Types

* **Boolean**: `true`, `false`
* **Integer**: Example: `42`, `-100`
* **Floating-point**: Example: `69.42`, `.1`, `6.`, `-3.14`
* **String**: `"Hello"`, `Hello` — Quotes can be omitted if the string doesn't conflict with a previously defined variable. Example: `"my string"`, `another_string_without_quotes`
* **Path**: Example: `./somewhere/relative.dat`, `/or/absolute`, `./../backneed/dotfirst.launch.py`
* **Array/Matrix**: `[ <Type>, <Type>, ... ]`, `[ 42, World, [ "nested" ] ]`, `[ [ 1, 2, 3 ], [ 4, 5, 6 ] ]` — Newlines after commas are also supported for readability. Example: `[1, 2, 3]`, `["apple", "banana"]`
* **Time**:
    * **Hour**: `hour`, `hours` (`s` is optional). Example: `2hours`, `1.5hour`
    * **Minute**: `min`, `mins` (`s` is optional). Example: `30min`, `5mins`
    * **Second**: `s`. Example: `10s`, `0.5s`
    * **Millisecond**: `ms`. Example: `200ms`, `1ms`
* **Length**:
    * **Meter**: `m`. Example: `10m`, `0.5m`
    * **Centimeter**: `cm`. Example: `50cm`, `2.5cm`
    * **Millimeter**: `mm`. Example: `100mm`, `1mm`
* **Range**:
    * **Time**: Example: `1ms..5hours`
    * **Length**: Example: `1mm..100m`

---

## Includes

In Ratslang, assigning a path to a namespace is not just intuitive; **it's how you perform an include operation**. This means the contents of the included file will be loaded into that specific namespace.

~~~awk
= ./path_relative_to_current_file.rl

strangefile.{
  = ./../namespacing_contents.rl
}
~~~

---

## Library Usage

First, you compile a Ratslang file to get a cleaned **Abstract Syntax Tree (AST)** with all variables resolved.

~~~rust
let ast = ratslang::compile_code("variable = true").unwrap();
let ast = ratslang::compile_file("./your_file.rl").unwrap();
~~~

Then, you can safely read the variables you need using Rust's powerful pattern matching.

~~~rust
// "Just get the variable"
let avar = ast.vars.resolve("password").unwrap();
let avar = ast.vars.resolve("_my_namespace._some_var").unwrap();
assert_eq!(avar, Some(Rhs::Val(Val::StringVal(":)".to_owned()))));

// Or truncate and filter namespaces first
let vars = ast.vars.filter_ns(&["_my_namespace"]);

// Match and handle unexpected types
let namespace = vars
      .resolve("_some_var")?
      .map_or(Ok("a_default_value".to_owned()), |rhs| {
          Ok(match rhs {
              rlc::Rhs::String(sval) => sval,
              _ => {
                  return Err(anyhow!("Unexpected type for _my_namespace._some_var, expected String."));
              }
          })
      })?;
~~~

---

## Extras

* Ratslang files typically use the `.rl` extension.
* Awk syntax highlighting provides a decent visual experience for Ratslang code.
* Compile errors are beautifully rendered thanks to [Ariadne](https://crates.io/crates/ariadne) ❤️.

A `crates.io` release with comprehensive documentation will be available soon, once a few more details are finalized.

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
