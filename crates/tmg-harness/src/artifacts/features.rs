//! `features.json` writer/reader.
//!
//! [`FeatureList`] manages the `features.json` artifact described in
//! SPEC §9.6 for harnessed runs. The schema is:
//!
//! ```json
//! {
//!   "features": [
//!     {
//!       "id": "feat-001",
//!       "category": "auth",
//!       "description": "User can log in with email",
//!       "steps": ["Visit /login", "Enter creds", "Reach /dashboard"],
//!       "passes": false
//!     }
//!   ]
//! }
//! ```
//!
//! ## Mutation policy
//!
//! The list of features is **immutable from the LLM's perspective**:
//! `id`, `category`, `description`, and `steps` are fixed at file
//! creation time (typically by the `initializer` subagent). The only
//! field the harness allows the LLM to change is `passes`, and only via
//! [`FeatureList::mark_passing`], which:
//!
//! - flips `passes` from `false` → `true` for an existing `id`,
//! - rejects unknown ids with [`HarnessError::UnknownFeatureId`],
//! - re-serialises and atomically writes back the file, preserving the
//!   exact ordering and every other field byte-for-byte (other than
//!   serde's pretty-printing of the targeted feature).
//!
//! Atomic write uses the same `*.tmp` + rename pattern as
//! [`RunStore::save`](crate::store::RunStore::save) and
//! [`SessionLog::save`](crate::artifacts::SessionLog::save), so concurrent
//! readers never observe a half-written file.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::HarnessError;

/// One feature entry in `features.json`.
///
/// Field ordering matches the SPEC §9.6 schema and is preserved on
/// round-trip through serde's preserve-order behaviour: each call to
/// [`FeatureList::mark_passing`] re-emits the entries in the order they
/// were read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feature {
    /// Unique identifier for this feature, e.g. `"feat-001"`.
    pub id: String,
    /// Free-form category label, e.g. `"auth"` or `"billing"`.
    pub category: String,
    /// Human-readable description.
    pub description: String,
    /// Reproduction steps as a list of short strings.
    pub steps: Vec<String>,
    /// Whether the feature has been verified as passing in this run.
    ///
    /// Initially `false`; flipped to `true` by
    /// [`FeatureList::mark_passing`].
    pub passes: bool,
}

/// Top-level structure of `features.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Features {
    /// Ordered list of features.
    pub features: Vec<Feature>,
}

/// Compact view of [`Features`] used by `session_bootstrap`.
///
/// Only the fields useful to the LLM at session start are retained:
/// `id`, `category`, and `passes`. The full text (`description`,
/// `steps`) is omitted to keep the bootstrap payload small. Use
/// [`FeatureList::read`] to obtain the full structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeaturesSummary {
    /// Total number of features in the file.
    pub total: usize,
    /// Number whose `passes` field is `true`.
    pub passing: usize,
    /// Up to `top_n` entries (taken in file order, passing first then
    /// failing) summarised by `id`, `category`, and `passes`.
    pub entries: Vec<FeaturesSummaryEntry>,
    /// Whether the original list contained more entries than `top_n`.
    pub truncated: bool,
}

/// One row in [`FeaturesSummary::entries`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeaturesSummaryEntry {
    /// Feature id.
    pub id: String,
    /// Feature category.
    pub category: String,
    /// Whether the feature is currently passing.
    pub passes: bool,
}

/// Append-only-ish writer/reader for a run's `features.json`.
///
/// Cloning a [`FeatureList`] is cheap (path-only).
#[derive(Debug, Clone)]
pub struct FeatureList {
    path: PathBuf,
}

impl FeatureList {
    /// Create a handle for the file at `path` without validating.
    ///
    /// Use [`load`](Self::load) if you want to validate the schema at
    /// construction time.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Load and validate the `features.json` schema at `path`.
    ///
    /// Returns a [`FeatureList`] handle on success. The file must
    /// already exist; this is a "harnessed-run-only" artifact created by
    /// the `initializer` subagent.
    ///
    /// # Errors
    ///
    /// - [`HarnessError::Io`] when the file cannot be read.
    /// - [`HarnessError::FeaturesDeserialize`] when the file is not
    ///   valid JSON or does not match the schema.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, HarnessError> {
        let path = path.as_ref().to_path_buf();
        let content = fs::read_to_string(&path).map_err(|e| HarnessError::io(&path, e))?;
        // Validate by parsing once; the parsed value is discarded so we
        // do not hold the entire structure in memory.
        let _: Features = serde_json::from_str(&content)
            .map_err(|e| HarnessError::features_deserialize(&path, e))?;
        Ok(Self { path })
    }

    /// Borrow the on-disk path of this features file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read and parse the full structure.
    pub fn read(&self) -> Result<Features, HarnessError> {
        let content =
            fs::read_to_string(&self.path).map_err(|e| HarnessError::io(&self.path, e))?;
        let parsed: Features = serde_json::from_str(&content)
            .map_err(|e| HarnessError::features_deserialize(&self.path, e))?;
        Ok(parsed)
    }

    /// Mark the feature with `feature_id` as passing.
    ///
    /// Behaviour:
    ///
    /// - Loads the existing file (validating its schema).
    /// - If `feature_id` is not present, returns
    ///   [`HarnessError::UnknownFeatureId`] without writing anything.
    /// - Otherwise sets `passes = true` for that id and atomically
    ///   writes the file back. Idempotent: marking an already-passing
    ///   feature is a no-op (the file is still re-written for
    ///   simplicity, but the contents remain equivalent).
    /// - All other fields (other features, category, description,
    ///   steps) are preserved exactly as read.
    pub fn mark_passing(&self, feature_id: &str) -> Result<(), HarnessError> {
        let mut parsed = self.read()?;
        let Some(target) = parsed.features.iter_mut().find(|f| f.id == feature_id) else {
            return Err(HarnessError::UnknownFeatureId {
                feature_id: feature_id.to_owned(),
            });
        };
        target.passes = true;
        self.write_atomic(&parsed)
    }

    /// Return `true` when every feature has `passes == true`. Empty
    /// feature lists are considered "all passing" (vacuously true).
    pub fn all_passing(&self) -> Result<bool, HarnessError> {
        let parsed = self.read()?;
        Ok(parsed.features.iter().all(|f| f.passes))
    }

    /// Return a [`FeaturesSummary`] containing up to `top_n` entries.
    ///
    /// Selection policy:
    /// 1. Failing features first (callers usually care about what is
    ///    not yet done);
    /// 2. then passing features, in file order;
    /// 3. truncated to `top_n` rows total.
    ///
    /// `top_n == 0` yields an empty `entries` list with `truncated`
    /// reflecting whether the file had any entries at all.
    pub fn summary(&self, top_n: usize) -> Result<FeaturesSummary, HarnessError> {
        let parsed = self.read()?;
        let total = parsed.features.len();
        let passing = parsed.features.iter().filter(|f| f.passes).count();

        let mut ordered: Vec<FeaturesSummaryEntry> = parsed
            .features
            .iter()
            .filter(|f| !f.passes)
            .map(FeaturesSummaryEntry::from)
            .collect();
        ordered.extend(
            parsed
                .features
                .iter()
                .filter(|f| f.passes)
                .map(FeaturesSummaryEntry::from),
        );

        let truncated = top_n < total;
        ordered.truncate(top_n);

        Ok(FeaturesSummary {
            total,
            passing,
            entries: ordered,
            truncated,
        })
    }

    /// Write `features` atomically (`*.tmp` + rename).
    fn write_atomic(&self, features: &Features) -> Result<(), HarnessError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| HarnessError::io(parent, e))?;
        }
        let tmp_path = self.path.with_extension("json.tmp");
        let serialized = serde_json::to_string_pretty(features)
            .map_err(|e| HarnessError::features_serialize(&self.path, e))?;

        if let Err(e) = fs::write(&tmp_path, &serialized) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&tmp_path, e));
        }
        if let Err(e) = fs::rename(&tmp_path, &self.path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(HarnessError::io(&self.path, e));
        }
        Ok(())
    }
}

impl From<&Feature> for FeaturesSummaryEntry {
    fn from(feature: &Feature) -> Self {
        Self {
            id: feature.id.clone(),
            category: feature.category.clone(),
            passes: feature.passes,
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic, reason = "test assertions use panic-based macros")]
mod tests {
    use super::*;

    fn sample_features_json() -> &'static str {
        r#"{
  "features": [
    {
      "id": "feat-001",
      "category": "auth",
      "description": "User can log in with email",
      "steps": ["Visit /login", "Enter creds", "Reach /dashboard"],
      "passes": false
    },
    {
      "id": "feat-002",
      "category": "billing",
      "description": "User can view invoices",
      "steps": ["Visit /invoices", "See list"],
      "passes": false
    },
    {
      "id": "feat-003",
      "category": "auth",
      "description": "User can log out",
      "steps": ["Click logout"],
      "passes": true
    }
  ]
}
"#
    }

    fn write_sample(content: &str) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let path = tmp.path().join("features.json");
        std::fs::write(&path, content).unwrap_or_else(|e| panic!("{e}"));
        (tmp, path)
    }

    #[test]
    fn load_validates_schema() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(list.path(), path);
    }

    #[test]
    fn load_rejects_invalid_json() {
        let (_tmp, path) = write_sample("not json");
        let result = FeatureList::load(&path);
        assert!(matches!(
            result,
            Err(HarnessError::FeaturesDeserialize { .. })
        ));
    }

    #[test]
    fn load_rejects_unknown_extra_fields() {
        let bad = r#"{
  "features": [
    {
      "id": "feat-001",
      "category": "auth",
      "description": "x",
      "steps": [],
      "passes": false,
      "evil_field": "smuggled"
    }
  ]
}"#;
        let (_tmp, path) = write_sample(bad);
        let result = FeatureList::load(&path);
        assert!(matches!(
            result,
            Err(HarnessError::FeaturesDeserialize { .. })
        ));
    }

    #[test]
    fn read_returns_full_structure() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let features = list.read().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(features.features.len(), 3);
        assert_eq!(features.features[0].id, "feat-001");
        assert_eq!(features.features[0].steps.len(), 3);
        assert!(features.features[2].passes);
    }

    #[test]
    fn mark_passing_flips_only_passes_field() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));

        let before = list.read().unwrap_or_else(|e| panic!("{e}"));
        list.mark_passing("feat-002")
            .unwrap_or_else(|e| panic!("{e}"));
        let after = list.read().unwrap_or_else(|e| panic!("{e}"));

        // Only the targeted feature's `passes` changed; every other
        // field is byte-equal.
        assert_eq!(before.features.len(), after.features.len());
        for (b, a) in before.features.iter().zip(after.features.iter()) {
            assert_eq!(b.id, a.id);
            assert_eq!(b.category, a.category);
            assert_eq!(b.description, a.description);
            assert_eq!(b.steps, a.steps);
            if b.id == "feat-002" {
                assert!(!b.passes);
                assert!(a.passes);
            } else {
                assert_eq!(b.passes, a.passes);
            }
        }
    }

    #[test]
    fn mark_passing_preserves_order() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));

        list.mark_passing("feat-001")
            .unwrap_or_else(|e| panic!("{e}"));
        let after = list.read().unwrap_or_else(|e| panic!("{e}"));
        let ids: Vec<&str> = after.features.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(ids, vec!["feat-001", "feat-002", "feat-003"]);
    }

    #[test]
    fn mark_passing_unknown_id_is_error() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let result = list.mark_passing("nonexistent");
        assert!(matches!(
            result,
            Err(HarnessError::UnknownFeatureId { ref feature_id }) if feature_id == "nonexistent"
        ));

        // The file must not have been modified after a rejected call.
        let after = list.read().unwrap_or_else(|e| panic!("{e}"));
        assert!(!after.features[0].passes);
        assert!(!after.features[1].passes);
        assert!(after.features[2].passes);
    }

    /// Schema invariance: the public API of `FeatureList` cannot be used
    /// to add, remove, or modify any field other than `passes`. This test
    /// exercises every public mutating method and asserts that the
    /// non-`passes` portion of the file is preserved exactly.
    #[test]
    fn mark_passing_cannot_add_or_remove_features() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));

        let before_ids: Vec<String> = list
            .read()
            .unwrap_or_else(|e| panic!("{e}"))
            .features
            .iter()
            .map(|f| f.id.clone())
            .collect();

        // Any number of mark_passing calls keep the id set unchanged.
        list.mark_passing("feat-001")
            .unwrap_or_else(|e| panic!("{e}"));
        list.mark_passing("feat-002")
            .unwrap_or_else(|e| panic!("{e}"));
        // Unknown id leaves the file untouched.
        let _ = list.mark_passing("not-real");
        // Idempotent on already-passing features.
        list.mark_passing("feat-003")
            .unwrap_or_else(|e| panic!("{e}"));

        let after_ids: Vec<String> = list
            .read()
            .unwrap_or_else(|e| panic!("{e}"))
            .features
            .iter()
            .map(|f| f.id.clone())
            .collect();
        assert_eq!(before_ids, after_ids);
    }

    /// If a malicious actor hand-edits the file to insert an unknown
    /// field while the harness is running, the next `mark_passing` call
    /// rejects the file rather than silently dropping the smuggled
    /// field.
    #[test]
    fn mark_passing_rejects_smuggled_extra_field_at_read_time() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));

        // Tamper with the file: add a wrapper-level extra field.
        let smuggled = r#"{
  "features": [
    {
      "id": "feat-001",
      "category": "auth",
      "description": "x",
      "steps": [],
      "passes": false
    }
  ],
  "evil": "smuggled"
}"#;
        std::fs::write(&path, smuggled).unwrap_or_else(|e| panic!("{e}"));

        let result = list.mark_passing("feat-001");
        assert!(matches!(
            result,
            Err(HarnessError::FeaturesDeserialize { .. })
        ));
    }

    #[test]
    fn all_passing_reports_state() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));

        assert!(!list.all_passing().unwrap_or_else(|e| panic!("{e}")));
        list.mark_passing("feat-001")
            .unwrap_or_else(|e| panic!("{e}"));
        list.mark_passing("feat-002")
            .unwrap_or_else(|e| panic!("{e}"));
        // feat-003 was already passing.
        assert!(list.all_passing().unwrap_or_else(|e| panic!("{e}")));
    }

    #[test]
    fn all_passing_on_empty_list_is_true() {
        let (_tmp, path) = write_sample(r#"{"features":[]}"#);
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(list.all_passing().unwrap_or_else(|e| panic!("{e}")));
    }

    #[test]
    fn summary_counts_and_orders_failing_first() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let summary = list.summary(10).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passing, 1);
        assert!(!summary.truncated);
        assert_eq!(summary.entries.len(), 3);
        // Failing first.
        assert_eq!(summary.entries[0].id, "feat-001");
        assert_eq!(summary.entries[1].id, "feat-002");
        // Passing last.
        assert_eq!(summary.entries[2].id, "feat-003");
        assert!(summary.entries[2].passes);
    }

    #[test]
    fn summary_truncates_at_top_n() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let summary = list.summary(2).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(summary.total, 3);
        assert_eq!(summary.entries.len(), 2);
        assert!(summary.truncated);
    }

    #[test]
    fn summary_zero_top_n_is_empty() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        let summary = list.summary(0).unwrap_or_else(|e| panic!("{e}"));
        assert!(summary.entries.is_empty());
        assert!(summary.truncated);
    }

    #[test]
    fn write_is_atomic_no_lingering_tmp() {
        let (_tmp, path) = write_sample(sample_features_json());
        let list = FeatureList::load(&path).unwrap_or_else(|e| panic!("{e}"));
        list.mark_passing("feat-001")
            .unwrap_or_else(|e| panic!("{e}"));
        let tmp_path = path.with_extension("json.tmp");
        assert!(!tmp_path.exists(), "atomic write should leave no tmp file");
    }
}
