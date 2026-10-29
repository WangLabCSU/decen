# Rust Import and Module Declaration Order

Organize the top of each file in this order:

1. Standard library imports (`std`, `core`, `alloc`).
2. Third-party imports, including third-party `pub use` statements.
3. Regular local module declarations (`mod name;`).
4. Local imports and re-exports (`use self::…`, `use super::…`, `use crate::…`,
   `pub use self::…`).
5. Test module declarations (`#[cfg(test)] mod tests;`).

Separate these groups with a blank line. Use explicit `self::` paths when
re-exporting items from local child modules.

Within each import group, place `pub use` statements before ordinary `use`
statements, with a blank line between them. Keep the standard library,
third-party, and local group order above.

Test module declarations must appear after all `use` and `pub use` statements.
Do not group them with regular module declarations before the imports. Inline
test modules may remain at the end of the file.

```rust
use std::io;

use external_crate::ExternalType;

mod component;

pub use self::component::Component;

use crate::Config;

#[cfg(test)]
mod tests;
```

# Documentation Requirements

- Apply these requirements throughout the project, not just to particular types
  or modules.
- Write API documentation for callers: explain what the operation does, what it
  returns, and how it changes observable state.
- Do not repeat guarantees inherent in the type's normal operation in every
  method's documentation. For example, storage-management methods do not need
  boilerplate about preserving data, its length, or previously split results, or
  vague statements such as "may move data." Focus on the operation's purpose,
  meaningful choices, and exceptions or failure effects callers need to know.
- Use the documentation of analogous standard library or established library
  APIs as a model. Describe this project's actual behavior; do not copy
  guarantees that do not apply.
- Use plain, concrete language. Explain parameter meanings, return values, and
  observable effects precisely. Prefer short before-and-after examples over
  abstract descriptions.
- Do not use the word `unread` in documentation. Describe the relevant data or
  state directly.
- Do not expose implementation details in public documentation unless they are
  necessary to use the API correctly. Internal fields, pointer manipulation,
  reference counting, initialization optimizations, and allocation or
  data-movement algorithms belong in ordinary code comments, not rustdoc.
- Describe behavior directly instead of translating assignments or branch
  conditions into prose. Avoid phrases such as "resets the position to zero" or
  "if the requested amount exceeds the available-space accessor" when callers
  only need to know that consumed bytes are discarded when making room. Linking
  to a public accessor does not make such implementation narration appropriate.
- Express relevant conditions in plain language, such as "if enough space is
  already available." Preserve meaningful guarantees and failure effects,
  including whether an unsuccessful operation can discard consumed bytes. Keep
  exact bounds or formulas only when callers need them to use the API correctly
  or safely.
- Document observable guarantees and necessary constraints, including relevant
  panic conditions, errors, ownership, borrowing, and safety requirements. Do
  not remove required `# Safety` documentation or omit facts callers need to use
  an API safely.
- Keep internal safety reasoning in `// SAFETY:` comments. Explain why the
  operation is valid, rather than merely asserting that it is safe; for example,
  explain why source and destination cannot overlap.
- When changing documentation, preserve code behavior and verify runnable
  examples with documentation tests. Keep examples short and focused on the
  operation being explained.
- When renaming an API, update related rustdoc, internal comments, README
  examples, and test names together. Use consistent terminology for the same
  operation throughout the project.

# Object Design Principle

- Do not make a field `pub` if doing so would allow callers to violate the
  object's invariants.
- Follow the ownership-boundary guidance in [Rust notes:
  `into_parts()`](https://github.com/Yunuuuu/rust-notes/blob/main/src/into_parts.md).
  Provide `into_parts(self)` only when downstream code actually needs ownership
  of fields without cloning large values such as `String` or `Vec`, and no
  longer needs the original object.

# Explicit Computational Costs

- Make computation and resource costs explicit in API semantics and at call
  sites. Allocation, copying, scanning, revalidation, encoding/decoding,
  normalization, and other processing should be explicit operations rather than
  hidden side effects of accessors, borrowing, or convenience conversions.
  Document their costs; a borrow or ownership move must not silently become a
  deep copy.
- Explicit exception: **implicit allocation or copying in a wrapper conversion
  is permitted when it comes from an existing underlying `Target: From<Source>`
  implementation provided by `std` or the crate defining the involved types.**
  Generic field/value-object wrappers may delegate component-wise to that
  implementation, including supported conversions from `&[u8]` or `&str` into
  owned byte or text storage such as `Vec<u8>`, `Box<[u8]>`, `String`, `Bytes`,
  or `BytesMut`. The wrapper must not introduce additional copying or simulate a
  missing underlying `From` implementation with `.to_vec()`, `.to_owned()`, or
  cloning. The corresponding `Into` calls are also permitted. Document the
  inherited copying costs and preserve the underlying implementation's lifetime
  and validity requirements.

# Commit Messages

- Prefix commit subjects with a change type, using `type: description` or
  `type(scope): description`.
- Choose the type that describes the change: `feat`, `fix`, `docs`, `test`,
  `refactor`, `perf`, `ci`, `build`, or `chore`.
- Write concise English descriptions of the actual change. For example,
  `feat: add framed decoding` or `ci: test stable and MSRV`.
