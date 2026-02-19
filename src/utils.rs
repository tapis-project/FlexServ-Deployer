//! Crate-wide utility functions (URL normalization, string predicates, ID normalization).

/// Returns true if `s` is a non-empty absolute HTTP or HTTPS URL (after trimming).
pub fn is_absolute_http_url(s: &str) -> bool {
    let s = s.trim();
    (s.starts_with("https://") || s.starts_with("http://")) && s.len() > 8
}

/// Normalize tenant URL: trim; if no scheme, prepend `https://`
/// (e.g. `tacc.tapis.io` → `https://tacc.tapis.io`).
pub fn normalize_tenant_url(url: &str) -> String {
    let s = url.trim();
    if s.is_empty() {
        return s.to_string();
    }
    if s.starts_with("https://") || s.starts_with("http://") {
        return s.to_string();
    }
    format!("https://{}", s)
}

/// Returns true if the string contains only ASCII lowercase letters and digits.
pub fn is_lowercase_alphanumeric(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
}

/// Normalize a string to lowercase ASCII alphanumeric only (e.g. strip dashes from a UUID).
/// Useful for deriving stable IDs from user input.
pub fn normalize_to_lowercase_alphanumeric(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_absolute_http_url() {
        assert!(is_absolute_http_url("https://tacc.tapis.io"));
        assert!(is_absolute_http_url("http://host"));
        assert!(is_absolute_http_url("  https://x.co  "));
        assert!(!is_absolute_http_url("tacc.tapis.io"));
        assert!(!is_absolute_http_url(""));
        assert!(!is_absolute_http_url("https://")); // len <= 8
    }

    #[test]
    fn test_normalize_tenant_url() {
        assert_eq!(normalize_tenant_url("tacc.tapis.io"), "https://tacc.tapis.io");
        assert_eq!(normalize_tenant_url("https://tacc.tapis.io"), "https://tacc.tapis.io");
        assert_eq!(normalize_tenant_url("  tacc.tapis.io  "), "https://tacc.tapis.io");
        assert_eq!(normalize_tenant_url(""), "");
    }

    #[test]
    fn test_is_lowercase_alphanumeric() {
        assert!(is_lowercase_alphanumeric(""));
        assert!(is_lowercase_alphanumeric("a"));
        assert!(is_lowercase_alphanumeric("abc123"));
        assert!(is_lowercase_alphanumeric("p1a2b3"));
        assert!(!is_lowercase_alphanumeric("aBc"));
        assert!(!is_lowercase_alphanumeric("Uppercase"));
        assert!(!is_lowercase_alphanumeric("with-dash"));
        assert!(!is_lowercase_alphanumeric("with_underscore"));
        assert!(!is_lowercase_alphanumeric("space "));
    }

    #[test]
    fn test_normalize_to_lowercase_alphanumeric() {
        assert_eq!(normalize_to_lowercase_alphanumeric(""), "");
        assert_eq!(normalize_to_lowercase_alphanumeric("550e8400-e29b-41d4-a716-446655440000"), "550e8400e29b41d4a716446655440000");
        assert_eq!(normalize_to_lowercase_alphanumeric("ABC-123"), "abc123");
    }
}
