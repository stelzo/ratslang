use core::fmt;
use std::{collections::HashMap, path::PathBuf, str::FromStr};

use anyhow::anyhow;
use ariadne::{Color, Label, Report, ReportKind, Source};
use logos::{Lexer, Logos};

use chumsky::{
    container::Seq,
    input::{Stream, ValueInput},
    prelude::*,
};

fn floating_millimeter<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let f: f64 = slice[..slice.len() - 2].parse().ok()?;
    Some(f)
}

fn floating_centimeter<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let f: f64 = slice[..slice.len() - 2].parse().ok()?;
    Some(f * 10.0)
}

fn floating_meter<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let f: f64 = slice[..slice.len() - 1].parse().ok()?;
    Some(f * 10.0 * 100.0)
}

fn integer_millimeter<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let f: i64 = slice[..slice.len() - 2].parse().ok()?;
    Some(f)
}

fn integer_centimeter<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let f: i64 = slice[..slice.len() - 2].parse().ok()?;
    Some(f * 10)
}

fn integer_meter<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let f: i64 = slice[..slice.len() - 1].parse().ok()?;
    Some(f * 10 * 100)
}

fn integer_second<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let f: i64 = slice[..slice.len() - 1].parse().ok()?;
    Some(f * 1000)
}

fn integer_minute<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let len = if slice.ends_with('s') { 4 } else { 3 };
    let f: i64 = slice[..slice.len() - len].parse().ok()?;
    Some(f * 60 * 1000)
}

fn integer_millisec<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let f: i64 = slice[..slice.len() - 2].parse().ok()?;
    Some(f)
}

fn integer_hour<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<i64> {
    let slice = lex.slice();
    let len = if slice.ends_with('s') { 5 } else { 4 };
    let f: i64 = slice[..slice.len() - len].parse().ok()?;
    Some(f * 60 * 60 * 1000)
}

fn floating_second<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let f: f64 = slice[..slice.len() - 1].parse().ok()?;
    Some(f * 1000.0)
}

fn floating_minute<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let len = if slice.ends_with('s') { 4 } else { 3 };
    let f: f64 = slice[..slice.len() - len].parse().ok()?;
    Some(f * 60.0 * 1000.0)
}

fn floating_millisec<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let f: f64 = slice[..slice.len() - 2].parse().ok()?;
    Some(f)
}

fn floating_hour<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    let slice = lex.slice();
    let len = if slice.ends_with('s') { 5 } else { 4 };
    let f: f64 = slice[..slice.len() - len].parse().ok()?;
    Some(f * 60.0 * 60.0 * 1000.0)
}

fn rm_first<'a>(lex: &mut Lexer<'a, Token<'a>>) -> &'a str {
    let slice = lex.slice();
    &slice[1..slice.len()]
}

fn rm_last<'a>(lex: &mut Lexer<'a, Token<'a>>) -> &'a str {
    let slice = lex.slice();
    &slice[0..slice.len() - 1]
}

fn rm_first_and_last<'a>(lex: &mut Lexer<'a, Token<'a>>) -> &'a str {
    let slice = lex.slice();
    &slice[1..slice.len() - 1]
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\f]+")]
enum Token<'a> {
    Error,

    // -- Control --
    #[token("+")]
    OpPlus,

    #[token("-")]
    OpMinus,

    // #[token("<")]
    // OpInclude,
    #[regex(".", priority = 1)]
    Dot,

    #[token(",")]
    Comma,

    #[regex(r"(#.*)?\n")]
    NewLine,

    #[token("(")]
    BracketOpen,

    #[token(")")]
    BracketClose,

    #[token("{")]
    BlockStart,

    #[token("}")]
    BlockEnd,

    #[token("[")]
    LParen,

    #[token("]")]
    RParen,

    // -- Operators --
    #[regex("=")]
    OpAssignToLeft,

    #[token("..")]
    OpRange,

    #[regex(r"\.?\/+[^ \n]*")]
    Path(&'a str),

    #[regex(r#""[^"]+""#, rm_first_and_last)]
    String(&'a str),

    // -- Keywords --
    #[token("if")]
    KwIf,

    #[token("~")]
    DoNotCareOptim, // TODO can not resolve paths anymore with home, right?

    // -- Expressions --
    #[token("true")]
    True,

    #[token("false")]
    False,

    #[regex(r"[+-]?\d+", |lex| lex.slice().parse().ok())]
    IntegerNumber(i64),

    #[regex(r"[+-]?(\d+)?\.\d*", |lex| lex.slice().parse().ok())]
    FloatingNumber(f64),

    #[regex(r"[+-]?(\d+)?\.\d*mm", floating_millimeter)]
    #[regex(r"[+-]?(\d+)?\.\d*m]", floating_meter)]
    #[regex(r"[+-]?(\d+)?\.\d*cm", floating_centimeter)]
    FloatingNumberMillimeter(f64),

    #[regex(r"[+-]?(\d+)?\.\d*s]", floating_second)]
    #[regex(r"[+-]?(\d+)?\.\d*ms]", floating_millisec)]
    #[regex(r"[+-]?(\d+)?\.\d*mins?]", floating_minute)]
    #[regex(r"[+-]?(\d+)?\.\d*hours?]", floating_hour)]
    FloatingNumberMillisecond(f64),

    #[regex(r"[+-]?(\d+cm)", integer_centimeter)]
    #[regex(r"[+-]?(\d+mm)", integer_millimeter)]
    #[regex(r"[+-]?(\d+m)", integer_meter)]
    IntegerNumberMillimeter(i64),

    #[regex(r"[+-]?(\d+s)", integer_second)]
    #[regex(r"[+-]?(\d+ms)", integer_millisec)]
    #[regex(r"[+-]?(\d+mins)", integer_minute)]
    #[regex(r"[+-]?(\d+hours)", integer_hour)]
    IntegerNumberMillisecond(i64),

    #[regex(r"[a-zA-Z_:]+\d*[a-zA-Z_:\d]*\.", rm_last)]
    VarNamespace(&'a str),

    #[regex(r"[a-zA-Z_:]+\d*[a-zA-Z_:\d]*")]
    Variable(&'a str),

    #[regex(r"_[a-zA-Z_:]+\d*[a-zA-Z_:\d]*", rm_first)]
    PredefVariable(&'a str),
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::VarNamespace(ns) => write!(f, "Namespace({:?})", ns),
            Self::Path(path) => write!(f, "Path({:?})", path),
            Self::String(string) => write!(f, "String({:?})", string),
            Self::Comma => write!(f, ","),
            Self::OpPlus => write!(f, "+"),
            Self::Dot => write!(f, "."),
            Self::OpMinus => write!(f, "-"),
            Self::Error => write!(f, "<error>"),
            Self::NewLine => write!(f, "\\n"),
            Self::BracketOpen => write!(f, "["),
            Self::BracketClose => write!(f, "]"),
            Self::BlockStart => write!(f, "{}", "{"),
            Self::BlockEnd => write!(f, "{}", "}"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::OpAssignToLeft => write!(f, "<-"),
            Self::OpRange => write!(f, ".."),
            Self::KwIf => write!(f, "if"),
            Self::DoNotCareOptim => write!(f, "~"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::IntegerNumber(s) => write!(f, "{}", s),
            Self::FloatingNumber(s) => write!(f, "{}", s),
            Self::FloatingNumberMillimeter(s) => write!(f, "{}mm", s),
            Self::FloatingNumberMillisecond(s) => write!(f, "{}ms", s),
            Self::IntegerNumberMillimeter(s) => write!(f, "{}mm", s),
            Self::IntegerNumberMillisecond(s) => write!(f, "{}ms", s),
            Self::Variable(s) => write!(f, "{}", s),
            Self::PredefVariable(s) => write!(f, "_{}", s),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NumVal {
    Floating(f64),
    Integer(i64),
}

impl PartialEq for NumVal {
    fn eq(&self, other: &Self) -> bool {
        let tolerance = 1e-6; // Adjust this value as needed
        match (self, other) {
            (NumVal::Floating(fl), NumVal::Floating(fr)) => (fl - fr).abs() < tolerance,
            (NumVal::Floating(_), NumVal::Integer(_)) => false,
            (NumVal::Integer(_), NumVal::Floating(_)) => false,
            (NumVal::Integer(il), NumVal::Integer(ir)) => il == ir,
        }
    }
}

impl Eq for NumVal {}

impl NumVal {
    pub fn integerize_unit_val(self) -> NumVal {
        if let NumVal::Floating(f) = self {
            NumVal::Integer(f.round() as i64)
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    TimeMilliseconds,
    WayMillimeter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Val {
    UnitedVal(UnitVal),
    NumVal(NumVal),
    StringVal(String),
    BoolVal(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitVal {
    pub val: u64,
    pub unit: Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rhs {
    Range { from: Val, to: Val },
    Var(Var),
    Path(String),
    // Expr() TODO evaluate so in the end it becomes atomic Rhs
    Val(Val),
    Array(Vec<Box<Self>>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Var {
    Log,
    Predef {
        name: String,
        namespace: Vec<String>,
    },
    User {
        name: String,
        namespace: Vec<String>,
    },
}

fn vec_prepend<T: Clone>(prepend: &[T], source: &[T]) -> Vec<T> {
    let mut out = prepend.to_vec();
    out.extend(source.iter().cloned());
    out
}

impl Var {
    pub fn add_namespace(self, ns: &Vec<String>) -> Self {
        match self {
            Var::User { name, namespace } => Var::User {
                name,
                namespace: vec_prepend(ns, &namespace),
            },
            Var::Log => Var::Log,
            Var::Predef { name, namespace } => Var::Predef {
                name,
                namespace: vec_prepend(ns, &namespace),
            },
        }
    }
}

impl FromStr for Var {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let splitted = s.split(".").collect::<Vec<_>>();

        let l = splitted.len();
        if l < 1 {
            return Err(anyhow!("Nothing to parse"));
        }

        let last = splitted[l - 1];

        Ok({
            let mut v: Vec<String> = vec![];
            if l > 1 {
                v.extend(splitted.into_iter().take(l - 1).map(String::from));
            }
            if let Some(predef) = last.strip_prefix("_") {
                Self::Predef {
                    name: predef.to_owned(),
                    namespace: v,
                }
            } else {
                Self::User {
                    name: last.to_owned(),
                    namespace: v,
                }
            }
        })
    }
}

#[derive(Debug)]
pub struct Evaluated {
    pub vars: VariableHistory,
}

impl Evaluated {
    pub fn new() -> Self {
        Self {
            vars: VariableHistory::new(vec![]),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Root {
    pub vardefs: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum StatementKind {
    VariableDef(Statement),
    VariableNamespaceBlock(Vec<Statement>),
}

#[derive(Debug, Clone)]
pub enum StatementKindPass1<'a> {
    VariableDef(Statement),
    Include {
        namespace: Vec<String>,
        path: String,
    },
    VariableNamespaceBlock {
        ns: Vec<&'a str>,
        stmts: Vec<Box<StatementKindPass1<'a>>>,
    },
}

#[derive(Debug, Clone)]
pub enum StatementKindOwnedPass1 {
    Include {
        namespace: Vec<String>,
        path: String,
    },
    VariableDef(Statement),
    VariableNamespaceBlock {
        ns: Vec<String>,
        stmts: Vec<Box<StatementKindOwnedPass1>>,
    },
}

#[derive(Debug, Clone)]
pub enum StatementKindOwned {
    VariableDef(Statement),
}

#[allow(unreachable_patterns)] // remove when adding more statments than just variables
fn do_pass1(
    pass1: Vec<StatementKindOwnedPass1>,
    current_parent: &std::path::Path,
) -> anyhow::Result<Vec<StatementKindOwned>> {
    let mut out = vec![];

    for stmt in pass1 {
        match stmt {
            StatementKindOwnedPass1::VariableDef(statement) => {
                out.push(StatementKindOwned::VariableDef(statement));
            }
            StatementKindOwnedPass1::VariableNamespaceBlock { ns, stmts } => {
                let v = stmts.into_iter().map(|f| *f).collect::<Vec<_>>();

                let passed = do_pass1(v, current_parent)?;
                out.extend(
                    passed
                        .into_iter()
                        .map(|t| match t {
                            StatementKindOwned::VariableDef(statement) => Ok(
                                StatementKindOwned::VariableDef(statement.add_namespace(&ns)),
                            ),
                            _ => Err(anyhow!(
                                "Only variable statements allowed inside a namespace block."
                            )),
                        })
                        .try_fold(Vec::new(), |mut acc, res| match res {
                            Ok(ok) => {
                                acc.push(ok);
                                Ok(acc)
                            }
                            Err(e) => Err(e),
                        })?,
                );
            }
            StatementKindOwnedPass1::Include { namespace, path } => {
                let to_parse = PathBuf::from_str(&path)?;
                let to_parse = if to_parse.is_relative() {
                    if to_parse.is_dir() {
                        return Err(anyhow!("File is a directory: {}", to_parse.display()));
                    } else {
                        current_parent.join(path).to_owned()
                    }
                } else {
                    to_parse
                };
                let nast = parse_to_ast(&to_parse)?;
                let nast = ast_add_namespace(&namespace, nast);
                out.extend(nast);
            }
        }
    }

    Ok(out)
}

fn parse_to_ast(path: &std::path::Path) -> anyhow::Result<Vec<StatementKindOwned>> {
    let file = std::fs::read_to_string(path);
    let file = match file {
        Ok(file) => Ok(file),
        Err(e) => match e.kind() {
            std::io::ErrorKind::NotFound => Err(anyhow!("Could not find file: {}", path.display())),
            std::io::ErrorKind::PermissionDenied => {
                Err(anyhow!("No permission to read file: {}", path.display()))
            }
            std::io::ErrorKind::IsADirectory => {
                Err(anyhow!("File is a directory: {}", path.display()))
            }
            _ => Err(anyhow!("Unexpected error for file: {}", path.display())),
        },
    }?;

    // parse file to be included recursively
    let token_iter = Token::lexer(&file).spanned().map(|(tok, span)| match tok {
        Ok(tok) => (tok, span.into()),
        Err(()) => (Token::Error, span.into()),
    });
    let token_stream =
        Stream::from_iter(token_iter).map((0..file.len()).into(), |(t, s): (_, _)| (t, s));
    match parser().parse(token_stream).into_result() {
        Ok(sexpr) => {
            let mut ast: Vec<StatementKindOwnedPass1> = Vec::new();
            for expr in sexpr {
                ast.push(expr.into());
            }

            let ast = do_pass1(
                ast,
                path.parent()
                    .ok_or(anyhow!("Can not get parent directory of file."))?,
            )?; // recursive call
            return Ok(ast);
        }
        Err(errs) => {
            for err in errs {
                Report::build(ReportKind::Error, (), err.span().start)
                    .with_code(3)
                    .with_message(err.to_string())
                    .with_label(
                        Label::new(err.span().into_range())
                            .with_message(err.reason().to_string())
                            .with_color(Color::Red),
                    )
                    .finish()
                    .eprint(Source::from(&file))
                    .unwrap();
            }
        }
    }

    return Err(anyhow!("Could not parse ratslang code."));
}

impl<'a> From<StatementKindPass1<'a>> for StatementKindOwnedPass1 {
    fn from(value: StatementKindPass1<'a>) -> Self {
        match value {
            StatementKindPass1::VariableNamespaceBlock { ns, stmts } => {
                StatementKindOwnedPass1::VariableNamespaceBlock {
                    ns: ns.into_iter().map(String::from).collect::<Vec<_>>(),
                    stmts: stmts
                        .into_iter()
                        .map(|bstmt| Box::new(StatementKindOwnedPass1::from(*bstmt)))
                        .collect::<Vec<_>>(),
                }
            }
            StatementKindPass1::VariableDef(statement) => {
                StatementKindOwnedPass1::VariableDef(statement)
            }
            StatementKindPass1::Include { namespace, path } => {
                StatementKindOwnedPass1::Include { namespace, path }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum StatementPass1 {
    AssignLeft {
        lhs: Vec<Var>,
        rhs: Rhs,
    },
    AssignRight {
        lhs: Rhs,
        rhs: Vec<Var>,
    },
    Include {
        namespace: Vec<String>,
        path: String,
    },
}

#[derive(Debug, Clone)]
pub enum Statement {
    AssignLeft { lhs: Vec<Var>, rhs: Rhs },
}

impl Statement {
    fn add_namespace(self, ns: &Vec<String>) -> Self {
        match self {
            Statement::AssignLeft { lhs, rhs } => {
                let nlhs = lhs
                    .into_iter()
                    .map(|v| v.add_namespace(&ns))
                    .collect::<Vec<_>>();
                Statement::AssignLeft { lhs: nlhs, rhs }
            }
        }
    }
}

#[allow(unreachable_patterns)] // remove when adding more statments than just variables
fn ast_add_namespace(ns: &Vec<String>, ast: Vec<StatementKindOwned>) -> Vec<StatementKindOwned> {
    ast.into_iter()
        .map(|stmt| match stmt {
            StatementKindOwned::VariableDef(statement) => {
                StatementKindOwned::VariableDef(statement.add_namespace(ns))
            }
            _ => stmt,
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub enum Operator {
    Left, // "<-"
}

// TODO replace with rhs
// #[derive(Clone, Copy, Debug)]
// pub enum Expr<'a> {
//     Var(&'a str),
//     Predef(&'a str),
//     Path(&'a str),
//     Log, // represents the LOG keyword.
// }

// This function signature looks complicated, but don't fear! We're just saying that this function is generic over
// inputs that:
//     - Can have tokens pulled out of them by-value, by cloning (`ValueInput`)
//     - Gives us access to slices of the original input (`SliceInput`)
//     - Produces tokens of type `Token`, the type we defined above (`Token = Token<'a>`)
//     - Produces spans of type `SimpleSpan`, a built-in span type provided by chumsky (`Span = SimpleSpan`)
// The function then returns a parser that:
//     - Has an input type of type `I`, the one we declared as a type parameter
//     - Produces an `SExpr` as its output
//     - Uses `Rich`, a built-in error type provided by chumsky, for error generation
fn parser<'a, I>()
-> impl Parser<'a, I, Vec<StatementKindPass1<'a>>, extra::Err<Rich<'a, Token<'a>>>>
where
    I: ValueInput<'a, Token = Token<'a>, Span = SimpleSpan>,
{
    let time = select! {
        Token::IntegerNumberMillisecond(ms) => NumVal::Integer(ms as i64),
        Token::FloatingNumberMillisecond(ms) => NumVal::Floating(ms).integerize_unit_val(),
    }
    .labelled("time");

    let way = select! {
        Token::IntegerNumberMillimeter(millis) => NumVal::Integer(millis as i64),
        Token::FloatingNumberMillimeter(millis) => NumVal::Floating(millis).integerize_unit_val(),
    }
    .labelled("way");

    let timerange = time
        .then_ignore(just(Token::OpRange))
        .then(time)
        .try_map(|(from, to), span| {
            Ok(Rhs::Range {
                from: {
                    Val::UnitedVal(UnitVal {
                        val: {
                            match from {
                                NumVal::Floating(_) => unreachable!("integerised before"),
                                NumVal::Integer(i) => {
                                    if i < 0 {
                                        return Err(Rich::custom(
                                            span,
                                            "Negative Integer Number for Timespan.",
                                        ));
                                    } else {
                                        i as u64
                                    }
                                }
                            }
                        },
                        unit: Unit::TimeMilliseconds,
                    })
                },
                to: Val::UnitedVal(UnitVal {
                    val: match to {
                        NumVal::Floating(_) => unreachable!("integerised before"),
                        NumVal::Integer(i) => {
                            if i < 0 {
                                return Err(Rich::custom(
                                    span,
                                    "Negative Integer Number for Timespan.",
                                ));
                            } else {
                                i as u64
                            }
                        }
                    },
                    unit: Unit::TimeMilliseconds,
                }),
            })
        })
        .labelled("time range");

    let wayrange = way
        .then_ignore(just(Token::OpRange))
        .then(way)
        .try_map(|(from, to), span| {
            Ok(Rhs::Range {
                from: Val::UnitedVal(UnitVal {
                    val: match from {
                        NumVal::Floating(_) => unreachable!("integerised before"),
                        NumVal::Integer(i) => {
                            if i < 0 {
                                return Err(Rich::custom(
                                    span,
                                    "Negative Integer Number for Wayspan.",
                                ));
                            } else {
                                i as u64
                            }
                        }
                    },
                    unit: Unit::WayMillimeter,
                }),
                to: Val::UnitedVal(UnitVal {
                    val: match to {
                        NumVal::Floating(_) => unreachable!("integerised before"),
                        NumVal::Integer(i) => {
                            if i < 0 {
                                return Err(Rich::custom(
                                    span,
                                    "Negative Integer Number for Wayspan.",
                                ));
                            } else {
                                i as u64
                            }
                        }
                    },
                    unit: Unit::WayMillimeter,
                }),
            })
        })
        .labelled("way range");

    let numvalue = select! {
            Token::IntegerNumber(i) => Rhs::Val(Val::NumVal(NumVal::Integer(i))),
            Token::FloatingNumber(f) => Rhs::Val(Val::NumVal(NumVal::Floating(f))),
    }
    .labelled("number value");

    let value = select! {
            Token::IntegerNumber(i) => Rhs::Val(Val::NumVal(NumVal::Integer(i))),
            Token::FloatingNumber(f) => Rhs::Val(Val::NumVal(NumVal::Floating(f))),
            Token::String(s) => Rhs::Val(Val::StringVal(s.to_owned())),
            Token::True => Rhs::Val(Val::BoolVal(true)),
            Token::False => Rhs::Val(Val::BoolVal(false)),
    }
    .labelled("value");

    let numberrange = numvalue
        .then_ignore(just(Token::OpRange))
        .then(value)
        .try_map(|(from, to), span| match (&from, &to) {
            (
                Rhs::Val(Val::NumVal(NumVal::Floating(_))),
                Rhs::Val(Val::NumVal(NumVal::Integer(_))),
            )
            | (
                Rhs::Val(Val::NumVal(NumVal::Integer(_))),
                Rhs::Val(Val::NumVal(NumVal::Floating(_))),
            ) => {
                return Err(Rich::custom(span, "Cannot span between float and integer."));
            }
            (_, _) => Ok((from, to)),
        })
        .map(|(from, to)| {
            let from = match from {
                Rhs::Val(Val::NumVal(nv)) => nv,
                _ => unreachable!(),
            };
            let to = match to {
                Rhs::Val(Val::NumVal(nv)) => nv,
                _ => unreachable!(),
            };

            Rhs::Range {
                from: Val::NumVal(from),
                to: Val::NumVal(to),
            }
        })
        .labelled("number range");

    let variable = select! {
            Token::VarNamespace(ns) => ns.to_owned(),
        }
        .repeated()
        .collect::<Vec<_>>()
        .then(select! {
                Token::Variable(var) => Rhs::Var(Var::User { name: var.to_owned(), namespace: vec![] }),
                Token::PredefVariable(var) => Rhs::Var(Var::Predef { name: var.to_owned(), namespace: vec![] }),
            })
        .map(|(ns, var)| {
                let var = match var {
                    Rhs::Var(var) => match var {
                        Var::User { name, namespace: _} => {
                            Var::User { name, namespace: ns }
                        },
                        Var::Predef { name, namespace: _ } => {
                            Var::Predef { name, namespace: ns }
                        },
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                Rhs::Var(var)
            });

    let term = select! {
        Token::Path(p) => Rhs::Path(p.to_owned()),
    }
    .or(variable)
    .labelled("term");

    let range = choice((numberrange, timerange.clone(), wayrange));

    let rhs_array = recursive(|rhs_array| {
        choice((term.clone(), range.clone(), value.clone(), rhs_array))
            .separated_by(just(Token::Comma).then_ignore(just(Token::NewLine).repeated()))
            .at_least(1)
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(|arr: Vec<Rhs>| {
                Rhs::Array(arr.into_iter().map(|r| Box::new(r)).collect::<Vec<_>>())
            })
    });

    let rhs = choice((term, range, value, rhs_array));

    // parse a comma-separated list of terms (the left-hand side).
    let comma = just(Token::Comma).then_ignore(just(Token::NewLine).repeated());
    let multi_rhs = rhs
        .clone()
        .separated_by(comma)
        .at_least(1)
        .collect::<Vec<_>>()
        .labelled("rule term");

    let op = select! {
        Token::OpAssignToLeft => Operator::Left,
    }
    .labelled("operator");

    // a statement is: lhs, then an operator, then a single term (the right-hand side).
    let statement = multi_rhs
        .clone()
        .then(op)
        .then(multi_rhs)
        .then_ignore(just(Token::NewLine).repeated())
        .try_map(|((lhs, op), rhs), span| {
            Ok(StatementKindPass1::VariableDef(match op {
                Operator::Left => {
                    if rhs.len() > 1 {
                        return Err(Rich::custom(
                            span,
                            "Can only assign one term but to multiple variables.",
                        ));
                    }
                    if let Some(rhs) = rhs.first() {
                        let mut vars: Vec<Var> = Vec::new();
                        for l in lhs.iter() {
                            match l {
                                Rhs::Var(var) => {
                                    vars.push(var.clone());
                                }
                                _ => {
                                    return Err(Rich::custom(
                                        span,
                                        "Can only assign to variables.",
                                    ));
                                }
                            }
                        }

                        Statement::AssignLeft {
                            lhs: vars,
                            rhs: rhs.clone(),
                        }
                    } else {
                        return Err(Rich::custom(span, "Missing right operand."));
                    }
                }
            }))
        })
        .labelled("statement");

    let include = just(Token::OpAssignToLeft)
        .ignore_then(select! {
            Token::Path(p) => p.to_owned(),
        })
        .then_ignore(just(Token::NewLine))
        .map(|path| StatementKindPass1::Include {
            namespace: vec![],
            path,
        });

    let namespace_block = recursive(|nsb| {
        select! {
            Token::VarNamespace(ns) => ns,
        }
        .repeated()
        .at_least(1)
        .collect()
        .then(
            choice((statement.clone(), include.clone(), nsb))
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(
                    just(Token::BlockStart).then_ignore(just(Token::NewLine).repeated()),
                    just(Token::BlockEnd).then_ignore(just(Token::NewLine).repeated()),
                ),
        )
        .map(|(ns, stmts): (Vec<&'_ str>, Vec<StatementKindPass1>)| {
            StatementKindPass1::VariableNamespaceBlock {
                ns,
                stmts: stmts
                    .into_iter()
                    .map(|stmtk| Box::new(stmtk))
                    .collect::<Vec<_>>(),
            }
        })
    });

    let one_of_wind_fns = choice((statement, namespace_block, include));
    let newlines = just(Token::NewLine).repeated();
    let one_block = newlines
        .clone()
        .ignore_then((one_of_wind_fns.then_ignore(newlines)).repeated().collect())
        .then_ignore(end());
    one_block
}

pub fn compile_file(
    path: &std::path::PathBuf,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> anyhow::Result<Evaluated> {
    compile_file_with_state(path, start_line, end_line, None, std::io::stderr(), true)
}

pub fn compile_file_with_state(
    path: &std::path::PathBuf,
    start_line: Option<usize>,
    end_line: Option<usize>,
    var_state: Option<HashMap<Var, Rhs>>,
    out: impl std::io::Write,
    rich_out: bool,
) -> anyhow::Result<Evaluated> {
    let file = std::fs::read_to_string(path.clone());
    let file = match file {
        Ok(file) => Ok(file),
        Err(e) => match e.kind() {
            std::io::ErrorKind::NotFound => Err(anyhow!("Could not find file: {}", path.display())),
            std::io::ErrorKind::PermissionDenied => {
                Err(anyhow!("No permission to read file: {}", path.display()))
            }
            std::io::ErrorKind::IsADirectory => {
                Err(anyhow!("File is a directory: {}", path.display()))
            }
            _ => Err(anyhow!("Unexpected error for file: {}", path.display())),
        },
    }?;
    let start_line = start_line.map(|l| l.saturating_sub(1)).unwrap_or_default();
    let lines = file.lines();
    let end_line = end_line.unwrap_or(lines.clone().count()).saturating_sub(1);
    let selected_lines = lines
        .skip(start_line)
        .take(end_line + 1 - start_line)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"; // append newline as funny hack to fix one-line problems

    let dir = path.parent().ok_or(anyhow!(
        "Could not get parent directory of source code file."
    ))?;
    compile_code_with_state(&selected_lines, dir, var_state, out, rich_out)
}

#[derive(Debug)]
pub struct VariableHistory {
    ast: Vec<StatementKindOwned>,
    pub var_cache: HashMap<Var, Rhs>,
}

fn is_sub_namespace(sub: &[String], super_namespace: &[String]) -> bool {
    if sub.len() > super_namespace.len() {
        return false;
    }

    if sub.len() == 0 {
        return false; // Empty is not a sub-namespace
    }

    if super_namespace.len() == 0 {
        return false;
    }

    for i in 0..sub.len() {
        if sub[i] != super_namespace[i] {
            return false;
        }
    }

    true
}

fn remove_prefix_from_target(prefix_definer: &[String], target_vec: &Vec<String>) -> Vec<String> {
    let prefix_len = prefix_definer.len();
    let target_len = target_vec.len();
    let compare_len = std::cmp::min(prefix_len, target_len);

    let mut common_prefix_len = 0;
    for i in 0..compare_len {
        if prefix_definer[i] == target_vec[i] {
            common_prefix_len += 1;
        } else {
            break;
        }
    }

    target_vec[common_prefix_len..].to_vec()
}

fn truncate_namespace_var(ns: &[String], var: &Var) -> Var {
    match var {
        Var::User { name, namespace } => Var::User {
            name: name.clone(),
            namespace: { remove_prefix_from_target(ns, namespace) },
        },
        Var::Log => Var::Log,
        Var::Predef { name, namespace } => Var::Predef {
            name: name.clone(),
            namespace: { remove_prefix_from_target(ns, namespace) },
        },
    }
}

fn truncate_namespace_rhs(ns: &[String], rhs: &Rhs) -> Rhs {
    match rhs {
        Rhs::Array(items) => Rhs::Array(
            items
                .iter()
                .map(|arr_item| Box::new(truncate_namespace_rhs(ns, arr_item)))
                .collect::<Vec<_>>(),
        ),
        Rhs::Range { from: _, to: _ } => rhs.clone(),
        Rhs::Var(var) => Rhs::Var(truncate_namespace_var(ns, var)),
        Rhs::Path(_) => rhs.clone(),
        Rhs::Val(_) => rhs.clone(),
    }
}

fn truncate_namespace_stmt(ns: &[String], stmt: &Statement) -> Statement {
    match stmt {
        Statement::AssignLeft { lhs, rhs } => {
            let l = lhs
                .iter()
                .filter(|v| match v {
                    Var::Log => false,
                    Var::Predef {
                        name: _,
                        namespace: var_ns,
                    } => is_sub_namespace(ns, &var_ns),
                    Var::User {
                        name: _,
                        namespace: var_ns,
                    } => is_sub_namespace(ns, &var_ns),
                })
                .map(|l| truncate_namespace_var(ns, l))
                .collect::<Vec<_>>();
            let r = truncate_namespace_rhs(ns, rhs);

            Statement::AssignLeft { lhs: l, rhs: r }
        }
    }
}

fn truncate_namespace(ns: &[String], stmt: &StatementKindOwned) -> StatementKindOwned {
    match stmt {
        StatementKindOwned::VariableDef(statement) => {
            StatementKindOwned::VariableDef(truncate_namespace_stmt(ns, statement))
        }
    }
}

fn stmt_in_ns(namespace: &[String], stmt: &Statement) -> bool {
    match stmt {
        Statement::AssignLeft { lhs, rhs: _ } => lhs.iter().any(|r| match r {
            Var::Log => false,
            Var::Predef {
                name: _,
                namespace: var_ns,
            } => is_sub_namespace(namespace, var_ns),
            Var::User {
                name: _,
                namespace: var_ns,
            } => is_sub_namespace(namespace, var_ns),
        }),
    }
}

impl VariableHistory {
    pub fn new(ast: Vec<StatementKindOwned>) -> Self {
        VariableHistory {
            ast,
            var_cache: HashMap::new(),
        }
    }

    pub fn with_state(mut self, state: HashMap<Var, Rhs>) -> Self {
        self.var_cache = state;
        self
    }

    #[allow(unreachable_patterns)] // remove when adding more statments than just variables
    pub fn populate_cache(&mut self) {
        self.ast.iter().for_each(|f| match f {
            StatementKindOwned::VariableDef(statement) => match statement {
                Statement::AssignLeft { lhs, rhs } => {
                    for l in lhs {
                        self.var_cache.insert(l.clone(), rhs.clone());
                    }
                }
            },
            _ => {}
        });
    }

    fn filter_by_namespace(
        &self,
        namespace: &[String],
        up_to: Option<usize>,
        trunc_ns: bool,
    ) -> Vec<StatementKindOwned> {
        let mut vars_in_ns = vec![];

        // use previous cached vars as well
        for (var, rhs) in self.var_cache.iter() {
            let mut wvar = var.clone();
            if is_sub_namespace(
                namespace,
                match var {
                    Var::Log => unreachable!(),
                    Var::Predef { name: _, namespace } => namespace,
                    Var::User { name: _, namespace } => namespace,
                },
            ) {
                if trunc_ns {
                    wvar = truncate_namespace_var(namespace, &wvar);
                }

                vars_in_ns.push(StatementKindOwned::VariableDef(Statement::AssignLeft {
                    lhs: vec![wvar],
                    rhs: rhs.clone(),
                }));
            }
        }

        for stmt in self.ast.iter().take(up_to.unwrap_or(usize::MAX)) {
            match &stmt {
                StatementKindOwned::VariableDef(rstmt) => {
                    if stmt_in_ns(namespace, rstmt) {
                        let rhs = trunc_ns
                            .then_some(truncate_namespace(namespace, stmt))
                            .unwrap_or(stmt.clone());
                        vars_in_ns.push(rhs);
                    }
                }
            }
        }

        vars_in_ns
    }

    fn resolve_recursive(&self, var: &Var, up_to: Option<usize>) -> anyhow::Result<Option<Rhs>> {
        // use previous state if already resolved there
        if let Some(cache_hit) = self.var_cache.get(var) {
            return Ok(Some(cache_hit.clone()));
        }

        let mut val = None;
        for (i, stmt) in self
            .ast
            .iter()
            .enumerate()
            .take(up_to.unwrap_or(usize::MAX))
        {
            match &stmt {
                &StatementKindOwned::VariableDef(statement) => match statement {
                    Statement::AssignLeft { lhs, rhs } => {
                        if lhs.contains(var) {
                            val = Some(match rhs {
                                Rhs::Var(var) => match self.resolve_recursive(&var, Some(i))? {
                                    None => Rhs::Val(Val::StringVal(match var {
                                        Var::User { name, namespace: _ } => name.clone(),
                                        Var::Log => {
                                            return Err(anyhow!("Log can not be assigned."));
                                        }
                                        Var::Predef {
                                            name: _,
                                            namespace: _,
                                        } => {
                                            return Err(anyhow!(
                                                "Predef variables can not be assigned."
                                            ));
                                        }
                                    })),
                                    Some(rhs) => rhs,
                                },
                                _ => rhs.clone(),
                            });
                        }
                    }
                },
            }
        }
        Ok(val)
    }

    fn prepend_ns(var: &Var, ns: &[String]) -> Var {
        match var {
            Var::Log => Var::Log,
            Var::Predef {
                name,
                namespace: var_ns,
            } => {
                let v = vec_prepend(ns, var_ns);

                Var::Predef {
                    name: name.clone(),
                    namespace: v,
                }
            }
            Var::User {
                name,
                namespace: var_ns,
            } => {
                let v = vec_prepend(ns, var_ns);

                Var::User {
                    name: name.clone(),
                    namespace: v,
                }
            }
        }
    }

    pub fn resolve_ns(&self, namespace: &[&str]) -> Vec<(Var, Rhs)> {
        let owned_ns = namespace
            .into_iter()
            .map(|s| (*s).to_owned())
            .collect::<Vec<_>>();
        self.filter_by_namespace(&owned_ns, None, true)
            .iter()
            .flat_map(|stmt| match stmt {
                StatementKindOwned::VariableDef(statement) => match statement {
                    Statement::AssignLeft { lhs, rhs: _ } => lhs
                        .iter()
                        .map(|lh| {
                            (
                                lh,
                                self.resolve_var(&Self::prepend_ns(lh, &owned_ns))
                                    .expect("Should have failed in Pass 2"),
                            )
                        })
                        .collect::<Vec<_>>(),
                },
            })
            .fold(Vec::new(), |mut acc, res| match res {
                (var, Some(rhs)) => {
                    acc.push((truncate_namespace_var(&owned_ns, var), rhs));
                    acc
                }
                (_, None) => acc, // skip if not set to anything
            })
    }

    pub fn filter_ns(&self, namespace: &[&str]) -> Self {
        let owned_ns = namespace
            .into_iter()
            .map(|s| (*s).to_owned())
            .collect::<Vec<_>>();
        let filtered = self.filter_by_namespace(&owned_ns, None, true);
        Self::new(filtered)
    }

    pub fn resolve(&self, var: &str) -> anyhow::Result<Option<Rhs>> {
        self.resolve_var(&Var::from_str(var)?)
    }

    fn resolve_var(&self, var: &Var) -> anyhow::Result<Option<Rhs>> {
        self.resolve_recursive(var, None)
    }
}

fn resolve_rhs_recursive(
    rhs: &mut Rhs,
    var_history: &VariableHistory,
    i: usize,
) -> anyhow::Result<()> {
    match rhs {
        Rhs::Array(items) => {
            for rhs_item in items.iter_mut() {
                resolve_rhs_recursive(&mut *rhs_item, var_history, i)?;
            }
        }
        Rhs::Var(var) => {
            // implicit strings when mentioned but not found in context earlier
            *rhs = match var_history.resolve_recursive(var, Some(i))? {
                None => Rhs::Val(Val::StringVal(match var {
                    Var::User { name, namespace: _ } => name.clone(),
                    Var::Log => {
                        return Err(anyhow!("Log can not be assigned."));
                    }
                    Var::Predef {
                        name: _,
                        namespace: _,
                    } => {
                        return Err(anyhow!("Predef variables can not be assigned."));
                    }
                })),
                Some(rhs) => rhs,
            };
        }
        _ => {}
    };

    Ok(())
}

fn resolve_stmt(
    stmt: &mut Statement,
    var_history: &VariableHistory,
    i: usize,
) -> anyhow::Result<()> {
    match stmt {
        Statement::AssignLeft { lhs: _, rhs } => {
            resolve_rhs_recursive(rhs, var_history, i)?;
        }
    }

    Ok(())
}

pub fn compile_code(source_code_raw: &str) -> anyhow::Result<Evaluated> {
    let currdir = std::env::current_dir()?;
    compile_code_with_state(
        &(source_code_raw.to_owned() + "\n"), // hack to fix one-line token problems
        &currdir,
        None,
        std::io::stderr(),
        true,
    )
}

pub fn compile_code_with_state(
    source_code_raw: &str,
    source_code_parent_dir: &std::path::Path,
    var_state: Option<HashMap<Var, Rhs>>,
    mut out: impl std::io::Write,
    rich_out: bool,
) -> anyhow::Result<Evaluated> {
    let token_iter = Token::lexer(&source_code_raw)
        .spanned()
        // Convert logos errors into tokens. We want parsing to be recoverable and not fail at the lexing stage, so
        // we have a dedicated `Token::Error` variant that represents a token error that was previously encountered
        .map(|(tok, span)| match tok {
            // Turn the `Range<usize>` spans logos gives us into chumsky's `SimpleSpan` via `Into`, because it's easier
            // to work with
            Ok(tok) => (tok, span.into()),
            Err(()) => (Token::Error, span.into()),
        });

    // Turn the token iterator into a stream that chumsky can use for things like backtracking
    let token_stream = Stream::from_iter(token_iter)
        // Tell chumsky to split the (Token, SimpleSpan) stream into its parts so that it can handle the spans for us
        // This involves giving chumsky an 'end of input' span: we just use a zero-width span at the end of the string
        .map((0..source_code_raw.len()).into(), |(t, s): (_, _)| (t, s));

    // Parse the token stream with our chumsky parser
    match parser().parse(token_stream).into_result() {
        Ok(sexpr) => {
            let owned = sexpr
                .into_iter()
                .map(|kind| kind.into())
                .collect::<Vec<StatementKindOwnedPass1>>();

            // Pass 1 -- parsing included files and expanding namespaces
            let mut ast = do_pass1(owned, source_code_parent_dir)?; // could print parse errors from included files

            // Pass 2 -- needs entire ast for resolving. TODO could still be combined to Pass 1 with a subset of the ast to t-1.
            // Resolve all variables here, else implicit strings in nested assigns are handled as variables.
            // Use the repl state in case the user already run some variables from before which we want to build upon
            let var_history =
                VariableHistory::new(ast.clone()).with_state(var_state.clone().unwrap_or_default());
            for (i, node) in ast.iter_mut().enumerate() {
                match node {
                    StatementKindOwned::VariableDef(statement) => {
                        resolve_stmt(statement, &var_history, i)?;
                    }
                }
            }

            let vars = VariableHistory::new(ast.clone()).with_state(var_state.unwrap_or_default());

            return Ok(Evaluated { vars });
        }
        Err(errs) => {
            let src = Source::from(source_code_raw);
            for err in errs {
                if rich_out {
                    let e = Report::build(ReportKind::Error, (), err.span().start)
                        .with_code(3)
                        .with_message(err.to_string())
                        .with_label(
                            Label::new(err.span().into_range())
                                .with_message(err.reason().to_string())
                                .with_color(Color::Red),
                        )
                        .finish();

                    e.write(src.clone(), &mut out).unwrap();
                } else {
                    let msg = format!(
                        "<{}-{}>  {}",
                        err.span().start,
                        err.span().end,
                        err.reason().to_string(),
                    );
                    out.write(msg.as_bytes()).unwrap();
                }
            }
        }
    }

    Err(anyhow!("Could not parse ratslang code."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        const SRC: &str = r"";
        let eval = compile_code(SRC);
        assert!(eval.is_ok());

        const SRC0: &str = r"
        ";
        let eval = compile_code(SRC0);
        assert!(eval.is_ok());

        const SRC1: &str = r"# blub";
        let eval = compile_code(SRC1);
        assert!(eval.is_ok());
    }

    #[test]
    fn comments() {
        const SRC: &str = r"
        # bla,.
        # ups /// \masd
        a = 1 # yo
        ";

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
    }

    #[test]
    fn var_declare() {
        const SRC: &str = r"
         a = 7
         b = a
         a = 6
         _l = /cloud
        ";

        // let mut a = Token::lexer(SRC);
        // let b = a.next();
        // dbg!(b);

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();

        let res = eval.vars.resolve("_l");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Path("/cloud".to_owned()));

        let res = eval.vars.resolve("a");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(6))));

        let res = eval.vars.resolve("b");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(7))));
    }

    #[test]
    fn var_declare_nested() {
        const SRC: &str = r"
         a.b.c = 7
         a.b = a.b.c
         a.b.c = 6
         wind._l = /cloud
        ";

        // let mut a = Token::lexer(SRC);
        // let b = a.next();
        // dbg!(b);

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();

        let res = eval.vars.resolve("wind._l");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Path("/cloud".to_owned()));

        let res = eval.vars.resolve("a.b.c");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(6))));

        let res = eval.vars.resolve("a.b");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(7))));
    }

    #[test]
    fn include() {
        const SRC: &str = r"
        # like assigning a file to the current namespace
        = ./test.rl

        # including in namespace blocks will prepend the namespaces to the included AST
        # (excluding rules, they are always global)
        c.{
            = ./test.rl
        }
        ";

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();

        let res = eval.vars.resolve("test");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(4))));

        let res = eval.vars.resolve("c.test");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(4))));
    }

    #[test]
    fn var_declare_blocks() {
        const SRC: &str = r"
        a.{
            b = 7
        }

        c.{
            t = 5

            d.{
                o = 1
            }
            i = 95
        }
        ";

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();

        let res = eval.vars.resolve("a.b");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(7))));

        let res = eval.vars.resolve("c.t");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(5))));

        let res = eval.vars.resolve("c.d.o");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(1))));

        let res = eval.vars.resolve("c.i");
        assert!(res.is_ok());
        let res = res.unwrap();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res, Rhs::Val(Val::NumVal(NumVal::Integer(95))));
    }

    #[test]
    fn in_ns() {
        // other, me
        let res = is_sub_namespace(&["a".to_owned()], &["a".to_owned()]); // true
        assert!(res);
        let res = is_sub_namespace(&["a".to_owned(), "b".to_owned()], &["a".to_owned()]); // a.b is deeper than a, so a.b can not be a sub namespace. false
        assert!(!res);
        let res = is_sub_namespace(&["a".to_owned()], &["a".to_owned(), "b".to_owned()]); // a is in a.b, true
        assert!(res);
        let res = is_sub_namespace(&[], &["a".to_owned()]); // empty is not in a.b, false
        assert!(!res);
    }

    #[test]
    fn resolve_ns() {
        const SRC: &str = r"
        _bag.{
            lidar.{
                short = l
                type = Pointcloud2
            }
            imu.{
                _short = i
                _type = Imu
            }
        }
        ";

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();
        let e = eval.vars.resolve_ns(&["_bag"]);
        assert_eq!(e.len(), 4);
    }

    #[test]
    fn strings() {
        const SRC: &str = r#"
        a = "bla with space"
        b = Blub
        c = Lidar::Ouster
            
        "#;

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();

        let e = eval.vars.resolve("a");
        assert!(e.is_ok());
        let e = e.unwrap();
        assert!(e.is_some());
        let rhs = e.unwrap();
        assert_eq!(rhs, Rhs::Val(Val::StringVal("bla with space".to_owned())));

        let e = eval.vars.resolve("b");
        assert!(e.is_ok());
        let e = e.unwrap();
        assert!(e.is_some());
        let rhs = e.unwrap();
        assert_eq!(rhs, Rhs::Val(Val::StringVal("Blub".to_owned())));

        let e = eval.vars.resolve("c");
        assert!(e.is_ok());
        let e = e.unwrap();
        assert!(e.is_some());
        let rhs = e.unwrap();
        assert_eq!(rhs, Rhs::Val(Val::StringVal("Lidar::Ouster".to_owned())));
    }

    #[test]
    fn arrays() {
        const SRC: &str = r#"
        a = [ "bla with space", Blub ]
        "#;

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();

        let e = eval.vars.resolve("a");
        assert!(e.is_ok());
        let e = e.unwrap();
        assert!(e.is_some());
        let rhs = e.unwrap();
        assert_eq!(
            rhs,
            Rhs::Array(vec![
                Box::new(Rhs::Val(Val::StringVal("bla with space".to_owned()))),
                Box::new(Rhs::Val(Val::StringVal("Blub".to_owned()))),
            ])
        );
    }

    // #[test]
    // fn enums() {
    //     // in your code
    //     #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    //     enum Lidar {
    //         Ouster,
    //         Sick,
    //     }

    //     #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    //     enum Sensor {
    //         Lidar(Lidar),
    //         Imu,
    //     }

    //     // for rl
    //     #[derive(Debug)]
    //     enum Enums {
    //         Sensor(Sensor),
    //     }
    //     let a = Enums::Sensor(Sensor::Lidar(Lidar::Ouster));
    //     let b = format!("{:?}", a);
    //     let r = convert_to_path(&b);
    //     println!("{:?}", r);
    //     const SRC: &str = "a = Sensor::Lidar";

    // TODO Give Enum to parser function that has all possible variants as trait. so fn variants(self--impl display) -> [impl FromStr]. Then call that in parser and pass type through to AST. If parser finds Sensor type, that has the same name as one of the enums in display. so it calls the parse function on that variant.
    #[test]
    fn vec_mat_vals() {
        const SRC: &str = r"
        vec = [3, 2, 4]

        mat = [ [3, 1, 1],
                [3, 2, 4] ]
        ";

        let eval = compile_code(SRC);
        assert!(eval.is_ok());
        let eval = eval.unwrap();
        let vec = eval.vars.resolve("vec").unwrap().unwrap();
        assert_eq!(
            vec,
            Rhs::Array(vec![
                Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(3)))),
                Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(2)))),
                Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(4)))),
            ])
        );
        let mat = eval.vars.resolve("mat").unwrap().unwrap();
        assert_eq!(
            mat,
            Rhs::Array(vec![
                Box::new(Rhs::Array(vec![
                    Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(3)))),
                    Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(1)))),
                    Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(1)))),
                ])),
                Box::new(Rhs::Array(vec![
                    Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(3)))),
                    Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(2)))),
                    Box::new(Rhs::Val(Val::NumVal(NumVal::Integer(4)))),
                ])),
            ])
        );
    }
}
