/// Events emitted by the streaming think parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ThinkEvent {
    /// Content that belongs to the thinking block.
    Thinking(String),
    /// Regular (non-thinking) content.
    Content(String),
}

/// Streaming state-machine parser for Gemma 4 thinking tags.
///
/// Open tag:  `<|channel>thought\n`
/// Close tag: `\n<channel|>`
///
/// When `include_thinking` is `false`, thinking tokens are silently discarded
/// and only [`ThinkEvent::Content`] events are ever emitted.
pub struct ThinkParser {
    /// Accumulated bytes that have not yet been emitted.
    buffer: String,
    /// Whether the parser is currently inside a thinking block.
    in_thinking: bool,
    /// Whether a thinking block has been seen at any point.
    thinking_started: bool,
    /// When `false`, thinking content is discarded instead of emitted.
    include_thinking: bool,
}

const THINK_OPEN: &str = "<|channel>thought\n";
const THINK_CLOSE: &str = "\n<channel|>";

impl ThinkParser {
    /// Create a new parser.
    ///
    /// * `include_thinking` – when `false`, thinking tokens are discarded.
    pub fn new(include_thinking: bool) -> Self {
        Self {
            buffer: String::new(),
            in_thinking: false,
            thinking_started: false,
            include_thinking,
        }
    }

    /// Feed one (or more) tokens and receive the events that can be determined
    /// without waiting for additional input.
    pub fn feed(&mut self, token: &str) -> Vec<ThinkEvent> {
        self.buffer.push_str(token);
        self.process()
    }

    /// Flush any remaining buffered text as a final event.
    ///
    /// Call this after the last token has been fed.
    pub fn flush(&mut self) -> Vec<ThinkEvent> {
        if self.buffer.is_empty() {
            return vec![];
        }
        let remaining = std::mem::take(&mut self.buffer);
        if self.in_thinking {
            if self.include_thinking {
                vec![ThinkEvent::Thinking(remaining)]
            } else {
                vec![]
            }
        } else {
            if remaining.is_empty() {
                vec![]
            } else {
                vec![ThinkEvent::Content(remaining)]
            }
        }
    }

    /// Whether a thinking block has been encountered during parsing.
    pub fn had_thinking(&self) -> bool {
        self.thinking_started
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Process as much of `self.buffer` as possible and return events.
    fn process(&mut self) -> Vec<ThinkEvent> {
        let mut events = Vec::new();

        loop {
            if self.in_thinking {
                // Look for the close tag.
                match find_or_prefix(&self.buffer, THINK_CLOSE) {
                    TagSearch::Found(pos) => {
                        // Emit thinking content up to the tag.
                        let thinking_content = self.buffer[..pos].to_string();
                        let after = self.buffer[pos + THINK_CLOSE.len()..].to_string();
                        self.buffer = after;
                        self.in_thinking = false;

                        if self.include_thinking && !thinking_content.is_empty() {
                            events.push(ThinkEvent::Thinking(thinking_content));
                        }
                        // Continue processing the remainder (could contain more
                        // open/close tags or plain content).
                    }
                    TagSearch::Prefix(prefix_len) => {
                        // Buffer ends with a prefix of THINK_CLOSE; we can
                        // safely emit everything before that prefix.
                        let safe_len = self.buffer.len() - prefix_len;
                        if safe_len > 0 {
                            let content = self.buffer[..safe_len].to_string();
                            self.buffer = self.buffer[safe_len..].to_string();
                            if self.include_thinking {
                                events.push(ThinkEvent::Thinking(content));
                            }
                        }
                        break; // Wait for more tokens.
                    }
                    TagSearch::NotFound => {
                        // No tag and no prefix – emit everything.
                        let content = std::mem::take(&mut self.buffer);
                        if self.include_thinking && !content.is_empty() {
                            events.push(ThinkEvent::Thinking(content));
                        }
                        break;
                    }
                }
            } else {
                // Look for the open tag.
                match find_or_prefix(&self.buffer, THINK_OPEN) {
                    TagSearch::Found(pos) => {
                        // Emit content before the tag.
                        if pos > 0 {
                            let before = self.buffer[..pos].to_string();
                            events.push(ThinkEvent::Content(before));
                        }
                        let after = self.buffer[pos + THINK_OPEN.len()..].to_string();
                        self.buffer = after;
                        self.in_thinking = true;
                        self.thinking_started = true;
                        // Continue to process remaining buffer.
                    }
                    TagSearch::Prefix(prefix_len) => {
                        // The tail of the buffer could be the start of an open
                        // tag; emit everything before that tail.
                        let safe_len = self.buffer.len() - prefix_len;
                        if safe_len > 0 {
                            let before = self.buffer[..safe_len].to_string();
                            self.buffer = self.buffer[safe_len..].to_string();
                            events.push(ThinkEvent::Content(before));
                        }
                        break; // Wait for more tokens.
                    }
                    TagSearch::NotFound => {
                        // No tag, no prefix – emit everything as content.
                        let content = std::mem::take(&mut self.buffer);
                        if !content.is_empty() {
                            events.push(ThinkEvent::Content(content));
                        }
                        break;
                    }
                }
            }
        }

        events
    }
}

// ---------------------------------------------------------------------------
// Tag-search helper
// ---------------------------------------------------------------------------

enum TagSearch {
    /// The tag was found at this byte offset.
    Found(usize),
    /// The tag was not found, but the last `n` bytes of the haystack are a
    /// prefix of the tag (so we must wait for more input).
    Prefix(usize),
    /// The tag was not found and no trailing prefix exists.
    NotFound,
}

/// Search `haystack` for `needle`.
///
/// Returns:
/// - `Found(pos)` if the needle appears in the haystack.
/// - `Prefix(n)` if no full match exists but the last `n` chars of the
///   haystack are a prefix of the needle (ambiguous – may complete later).
/// - `NotFound` otherwise.
fn find_or_prefix(haystack: &str, needle: &str) -> TagSearch {
    if let Some(pos) = haystack.find(needle) {
        return TagSearch::Found(pos);
    }

    // Check whether any suffix of `haystack` is a prefix of `needle`.
    // We walk from the longest possible overlap down to 1.
    // Only slice at valid UTF-8 char boundaries to avoid panics on multi-byte chars.
    let max_overlap = needle.len().min(haystack.len());
    for overlap in (1..=max_overlap).rev() {
        let start = haystack.len() - overlap;
        if !haystack.is_char_boundary(start) || !needle.is_char_boundary(overlap) {
            continue;
        }
        let haystack_tail = &haystack[start..];
        let needle_head = &needle[..overlap];
        if haystack_tail == needle_head {
            return TagSearch::Prefix(overlap);
        }
    }

    TagSearch::NotFound
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_thinking() {
        let mut parser = ThinkParser::new(true);
        let events = parser.feed("Hello world");
        assert_eq!(events, vec![ThinkEvent::Content("Hello world".into())]);
    }

    #[test]
    fn test_thinking_then_content() {
        let mut parser = ThinkParser::new(true);
        let mut all = Vec::new();
        all.extend(parser.feed("<|channel>thought\n"));
        all.extend(parser.feed("Let me think..."));
        all.extend(parser.feed("\n<channel|>"));
        all.extend(parser.feed("The answer is 4."));
        all.extend(parser.flush());
        assert!(all.iter().any(|e| matches!(e, ThinkEvent::Thinking(_))));
        assert!(all.iter().any(|e| matches!(e, ThinkEvent::Content(_))));
    }

    #[test]
    fn test_thinking_stripped_when_disabled() {
        let mut parser = ThinkParser::new(false);
        let mut all = Vec::new();
        all.extend(parser.feed("<|channel>thought\nSecret\n<channel|>Visible"));
        all.extend(parser.flush());
        assert!(all.iter().all(|e| matches!(e, ThinkEvent::Content(_))));
        let text: String = all
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Content(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("Visible"));
        assert!(!text.contains("Secret"));
    }

    #[test]
    fn test_had_thinking_false_when_none() {
        let mut parser = ThinkParser::new(true);
        parser.feed("Just plain text.");
        parser.flush();
        assert!(!parser.had_thinking());
    }

    #[test]
    fn test_had_thinking_true_when_present() {
        let mut parser = ThinkParser::new(true);
        parser.feed("<|channel>thought\nI am thinking.\n<channel|>Done.");
        parser.flush();
        assert!(parser.had_thinking());
    }

    #[test]
    fn test_partial_open_tag_held_in_buffer() {
        // Feed only a prefix of the open tag – nothing should be emitted yet.
        let mut parser = ThinkParser::new(true);
        let events = parser.feed("<|channel>tho");
        assert!(events.is_empty(), "expected no events for partial open tag");
    }

    #[test]
    fn test_partial_open_tag_then_no_match_flushes_content() {
        // Feed a prefix that turns out NOT to be the open tag.
        let mut parser = ThinkParser::new(true);
        let mut all = Vec::new();
        // "<|channel>tho" looks like a prefix of THINK_OPEN …
        all.extend(parser.feed("<|channel>tho"));
        // … but now we add something that proves it isn't.
        all.extend(parser.feed("se are not thinking tags"));
        all.extend(parser.flush());
        let text: String = all
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Content(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("<|channel>tho"));
    }

    #[test]
    fn test_content_before_thinking_emitted() {
        let mut parser = ThinkParser::new(true);
        let mut all = Vec::new();
        all.extend(parser.feed("Preamble. <|channel>thought\nthought\n<channel|>Suffix."));
        all.extend(parser.flush());
        let content: String = all
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Content(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(content.contains("Preamble."));
        assert!(content.contains("Suffix."));
    }

    #[test]
    fn test_flush_empty() {
        let mut parser = ThinkParser::new(true);
        let events = parser.flush();
        assert!(events.is_empty());
    }

    #[test]
    fn test_multiple_thinking_blocks() {
        let mut parser = ThinkParser::new(true);
        let mut all = Vec::new();
        all.extend(parser.feed("<|channel>thought\nFirst thought\n<channel|>Between.<|channel>thought\nSecond thought\n<channel|>End."));
        all.extend(parser.flush());
        let thinking: Vec<&str> = all
            .iter()
            .filter_map(|e| match e {
                ThinkEvent::Thinking(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking.len(), 2);
        assert!(thinking[0].contains("First thought"));
        assert!(thinking[1].contains("Second thought"));
    }
}
