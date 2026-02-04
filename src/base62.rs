//! A simple library base62 encode/decode, no dependencies other libraries.
// from : https://github.com/hongweipeng/rust-base62/blob/main/src/lib.rs
// standard 62-encoding, with a 32-byte input block and, a
// 43-byte output block.
const BASE256BLOCK_LEN: usize = 32;
const BASE62BLOCK_LEN: usize = 43;
const BASE62_LOG2: f64 = 5.954196310386875; // the result of `62f64.log2()`

const ALPHABET_SIZE: usize = 62;

const ALPHABET: [char; ALPHABET_SIZE] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
    'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z',
];

/*
ALPHABET_VERT: [u8; 256] = [0xff: 256];
for (i, &v) in ALPHABET.iter().enumerate() {
    ALPHABET_VERT[v as usize] = i as u8;
}
 */
const ALPHABET_VERT: [u8; 256] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 255, 255, 255,
    255, 255, 255, 255, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 255, 255, 255, 255, 255, 255, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

fn encode_len(n: usize) -> usize {
    if n == BASE256BLOCK_LEN {
        return BASE62BLOCK_LEN;
    }
    let n_block = n / BASE256BLOCK_LEN;
    let mut out = n_block * BASE62BLOCK_LEN;
    let rem = n % BASE256BLOCK_LEN;
    if rem > 0 {
        out += ((rem * 8) as f64 / BASE62_LOG2).ceil() as usize;
    }
    out
}

fn decode_len(n: usize) -> usize {
    let n_block = n / BASE62BLOCK_LEN;
    let mut out = n_block * BASE256BLOCK_LEN;
    let rem = n % BASE62BLOCK_LEN;
    if rem > 0 {
        out += (rem as f64 * BASE62_LOG2 / 8f64).floor() as usize;
    }
    out
}

fn is_valid_encoding_length(n: usize) -> bool {
    fn f(x: usize) -> usize {
        ((x as f64) * BASE62_LOG2 / 8f64).floor() as usize
    }
    f(n) != f(n - 1)
}

/// Encode `bytes` using the base62, return `String`.
pub fn encode(src: &[u8]) -> String {
    if src.is_empty() {
        return "".to_string();
    }
    let mut rs: usize = 0;
    let cap = encode_len(src.len());
    let mut dst = vec![0u8; cap];
    for b in src.iter().copied() {
        let mut c: usize = 0;
        let mut carry = b as usize;
        for j in (0..cap).rev() {
            if carry == 0 && c >= rs {
                break;
            }
            carry += 256 * dst[j] as usize;
            dst[j] = (carry % ALPHABET_SIZE) as u8;
            carry /= ALPHABET_SIZE;
            c += 1;
        }
        rs = c;
    }
    dst.iter().map(|&i| ALPHABET[i as usize]).collect()
}

#[derive(Debug)]
pub enum Error {
    BadInput { reason: String },
}

/// Decode `bytes` using the base62, return `Result<Vec<u8>, Error>`.
pub fn decode(src: &[u8]) -> Result<Vec<u8>, Error> {
    if src.is_empty() {
        return Ok(vec![]);
    }
    if !is_valid_encoding_length(src.len()) {
        return Err(Error::BadInput {
            reason: "invalid input length".to_string(),
        });
    }
    let mut rs: usize = 0;
    let cap = decode_len(src.len());
    let mut dst = vec![0u8; cap];
    for b in src.iter().copied() {
        let mut c: usize = 0;
        let mut carry: usize = ALPHABET_VERT[b as usize] as usize;
        if carry == 255 {
            return Err(Error::BadInput {
                reason: format!("bad input {}", b),
            });
        }
        for j in (0..cap).rev() {
            if carry == 0 && c >= rs {
                break;
            }
            carry += ALPHABET_SIZE * (dst[j] as usize);
            dst[j] = (carry % 256) as u8;
            carry /= 256;
            c += 1;
        }
        rs = c;
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_bytes(plain: &[u8], cipher: &[u8]) {
        assert_eq!(cipher, encode(plain).as_bytes());
        let result = decode(cipher);
        assert!(result.is_ok());
        assert_eq!(plain, result.unwrap());
    }
    fn check_str(plaintext: &str, ciphertext: &str) {
        check_bytes(plaintext.as_bytes(), ciphertext.as_bytes());
    }

    #[test]
    fn test_str() {
        check_str("", "");
        check_str("f", "1e");
        check_str("fo", "6ox");
        check_str("foo", "0SAPP");
        check_str("foob", "1sIyuo");
        check_str("fooba", "7kENWa1");
        check_str("foobar", "0VytN8Wjy");

        check_str("su", "7gj");
        check_str("sur", "0VkRe");
        check_str("sure", "275mAn");
        check_str("sure.", "8jHquZ4");
        check_str("asure.", "0UQPPAab8");
        check_str("easure.", "26h8PlupSA");
        check_str("leasure.", "9IzLUOIY2fe");

        check_str("Hello, World!", "1wJfrzvdbtXUOlUjUf");
        check_str("你好，世界！", "1ugmIChyMAcCbDRpROpAtpXdp");
        check_str("こんにちは", "1fyB0pNlcVqP3tfXZ1FmB");
        check_str("안녕하십니까", "1yl6dfHPaO9hroEXU9qFioFhM");

        check_str("=", "0z");
        check_str(">", "10");
        check_str("?", "11");
        check_str("11", "3H7");
        check_str("111", "0DWfh");
        check_str("1111", "0tquAL");
        check_str("11111", "3icRuhV");
        check_str("111111", "0FMElG7cn");
        check_str(
            "333333333333333333333333333333333333333",
            "12crJoybWfE2zqqnxPeYnbDOEcx8Lkv7ksPxzAA8kmM5Yb25Eb6bD",
        );
    }

    #[test]
    fn test_large_text() {
        // big text
        let s = "3333333333333".repeat(900);
        let e = encode(&s.as_bytes());
        let r = decode(e.as_bytes()).unwrap();
        assert_eq!(s, String::from_utf8(r).unwrap());
    }

    #[test]
    fn test_integer() {
        {
            // zero
            check_bytes(&[], "".as_bytes());
            check_bytes(&[0], "00".as_bytes());
            check_bytes(&[0, 0], "000".as_bytes());
            check_bytes(&[0, 0, 0], "00000".as_bytes());
            check_bytes(&[0, 0, 0, 0], "000000".as_bytes());
            check_bytes(&[0; 1025], "0".repeat(1378).as_bytes());

            // leading zero
            check_bytes(&[1], "01".as_bytes());
            check_bytes(&[2], "02".as_bytes());
            check_bytes(&[61], "0z".as_bytes());
            check_bytes(&[62], "10".as_bytes());
            check_bytes(&[100], "1c".as_bytes());
            check_bytes(&[0, 1], "001".as_bytes());
            check_bytes(&[0, 0, 0, 5], "000005".as_bytes());
            check_bytes(&[0, 0, 0, 0, 0, 62], "000000010".as_bytes());
        }
        {
            let bytes = (u64::MAX).to_be_bytes();
            check_bytes(&bytes, "LygHa16AHYF".as_bytes());

            let bytes = (u64::MAX as u128 + 1).to_be_bytes(); // exist leading zero
            check_bytes(&bytes, "00000000000LygHa16AHYG".as_bytes());
        }
        {
            let bytes = (ALPHABET_SIZE as u128).pow(21).to_be_bytes();
            check_bytes(&bytes, "1000000000000000000000".as_bytes());

            let bytes = (ALPHABET_SIZE as u128).pow(20).to_be_bytes();
            check_bytes(&bytes, "0100000000000000000000".as_bytes());

            let bytes = 92202686130861137968548313400401640448_u128.to_be_bytes();
            check_bytes(&bytes, "26tF05fvSIgh0000000000".as_bytes());
        }
    }

    #[test]
    fn test_invalid() {
        assert!(decode(&[1, 2, 3]).is_err());
        assert!(decode("73XpUgzMGA-jX6SV".as_bytes()).is_err());
    }
}
