use once_cell::sync::Lazy;
use regex::Regex;
use std::{cmp::Ordering, collections::HashMap, vec};

const R50K_PAT: &str = r"(?x)
    '(?:[sdmt]|ll|ve|re)
    | ?\p{L}+
    | ?\p{N}+
    | ?[^\s\p{L}\p{N}]+
    |\s+$
    |\s+(?!\S)
    |\s
";

const TOKEN_ID: usize = 256;

static R50K_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(R50K_PAT).unwrap());

struct BPETokenizer {
    pre_token_frequency: HashMap<Vec<u32>, usize>,
    frequency: HashMap<(u32, u32), usize>,
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u32>,
    token_to_pre_token: HashMap<u32, Vec<Vec<u32>>>,
    vocab_size: usize,
}

impl BPETokenizer {
    fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::with_capacity(256);

        for i in 0u8..=255 {
            id_to_token.push(vec![i]);
            token_to_id.insert(vec![i], i as u32);
        }

        BPETokenizer {
            pre_token_frequency: HashMap::new(),
            frequency:HashMap::new(),
            id_to_token,
            token_to_id,
            token_to_pre_token: HashMap::new(),
            vocab_size: 5000,
        }
    }


    fn train(&mut self, text: &str) {
        let tokens = R50K_REGEX
            .find_iter(text)
            .map(|t| t.as_str().as_bytes())
            .collect::<Vec<&[u8]>>();
        for token in tokens {
            let token_ids:Vec<u32> = token.iter().map(|&c| c as u32).collect();
            *self.pre_token_frequency.entry(token_ids).or_default() += 1;
        }
        for (k, v) in self.pre_token_frequency.iter() {
                for i in 1..k.len() {
                let pair = (k[i-1], k[i]);
                *self.frequency.entry(pair).or_default() += v
            }
        }
        while self.id_to_token.len() < self.vocab_size && self.frequency.len() > 1 {
            self.merge()
        }
    }

    fn merge(&mut self) {
        if let Some((&pair, count)) = self.frequency.iter().max_by(|x, y| {
            match y.1.cmp(x.1) {
                Ordering::Equal => y.0.cmp(x.0),
                other => other
            }
        }) {
            let (a, b) = pair;
            let mut merged = self.id_to_token[a as usize].clone();
            merged.extend_from_slice(&self.id_to_token[b as usize]);

            let new_id: u32 = if let Some(&existing_id) = self.token_to_id.get(&merged) {
                existing_id
            } else {
                let id = self.id_to_token.len() as u32;
                self.id_to_token.push(merged.clone());
                self.token_to_id.insert(merged.clone(), id);
                id
            };

            let mut new_pre_tokens = HashMap::new();

            for (t, &f) in self.pre_token_frequency.iter() {
                let mut res = Vec::with_capacity(t.len());

                let mut i = 0usize;
                while i < t.len() {
                    if i + 1 < t.len() && (t[i], t[i + 1]) == pair {
                        res.push(new_id);
                        i += 2;
                    } else {
                        res.push(t[i]);
                        i += 1;
                    }
                }
                *new_pre_tokens.entry(res).or_default() += f;
            }

            self.pre_token_frequency = new_pre_tokens;
            self.frequency.clear();
            for (tokens, &count) in &self.pre_token_frequency {
                for i in 1..tokens.len() {
                    let pair = (tokens[i - 1], tokens[i]);
                    *self.frequency.entry(pair).or_default() += count;
                }
            }
        }
    }


    fn encode(&self, text: &str) -> Vec<u32> {
        let mut output = Vec::new();

        let pre_tokens = R50K_REGEX
            .find_iter(text)
            .map(|t| t.as_str().as_bytes().to_vec());

        for token in pre_tokens {
            let mut ids: Vec<u32> = token.iter().map(|&b| b as u32).collect();

            loop {
                let mut merged = false;
                let mut i = 0;
                let mut new_ids = Vec::new();

                while i < ids.len() {
                    if i + 1 < ids.len() {
                        let pair = (ids[i], ids[i + 1]);
                        let mut merged_bytes = self.id_to_token[pair.0 as usize].clone();
                        merged_bytes.extend_from_slice(&self.id_to_token[pair.1 as usize]);

                        if let Some(&new_id) = self.token_to_id.get(&merged_bytes) {
                            new_ids.push(new_id);
                            i += 2;
                            merged = true;
                            continue;
                        }
                    }
                    new_ids.push(ids[i]);
                    i += 1;
                }

                ids = new_ids;
                if !merged {
                    break;
                }
            }

            output.extend(ids);
        }

        output
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.id_to_token.get(id as usize) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

