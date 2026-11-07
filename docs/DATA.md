# Data adapters and storage independence

The internal interface is a table of `path`, `label`, globally unique `id`, optional `group`, `cohort`, and optional predefined `role`. This is an adapter interface, not a required on-disk dataset layout. No clinical variable is used by the classifiers.

## Tables

Supported: CSV, TSV, Excel (`.xlsx`; legacy `.xls` additionally needs a pandas-compatible engine such as xlrd), JSON record arrays and JSONL. Each source has its own column names, root and cohort. Paths in the table are relative to that source's configured `root`; the root/table paths themselves are relative to the YAML. Absolute paths and environment variables also work. UTF-8 filenames and spaces are supported.

```yaml
data:
  label_map: {effusion: fluid, healthy: normal}
  sources:
    - adapter: table
      table: ../my_metadata.xlsx
      sheet: patients
      root: ../images_any_layout
      columns:
        path: image_filename
        label: reference_diagnosis
        id: examination_key
        group: patient_key
      cohort: internal
      id_prefix: 'hospital_a:'
    - adapter: table
      table: ../validation_table.jsonl
      root: ../independent_storage
      columns: {path: resource, label: truth, id: image_key}
      cohort: hospital_b
      id_prefix: 'hospital_b:'
```

Quote prefixes containing a colon in an actual YAML (`id_prefix: 'hospital_a:'`). IDs may be generated from path/cohort/label if omitted, but repeated file records need explicit unique IDs. CSV IDs are read as strings to preserve leading zeros. `group_prefix` is available when sites have different patients with the same local patient numbers; do not namespace known shared patients just to hide overlap. `defaults` adds missing canonical fields, including a constant reference label for a single-class source.

## Scan folders without moving files

```yaml
data:
  sources:
    - adapter: folders
      root: ../existing_collection
      glob: '**/*'
      label_parent: 0  # immediate parent directory is the label
      cohort: internal
```

For nested names, use `label_regex` with a named `(?P<label>...)` group on each relative path. Example: `'visits/[^/]+/(?P<label>[^/]+)/'`. A mapping can then translate source labels. Scanning is optional; readers can keep all images together and supply labels in a table instead.

## Custom metadata and image readers

```yaml
data:
  sources:
    - adapter: custom
      factory: examples.custom_adapter:records
      metadata_file: ../my_database_export.json
      loader: examples.custom_adapter:load_image
```

The factory receives its source configuration plus `_config_dir` and returns records/DataFrame with canonical fields. The `loader` receives one record as a dictionary and returns a PIL RGB image or a uint8 H×W×3 array. When a custom loader is supplied, `path` may be a resource key (archive member, database ID, DICOM reference, etc.) instead of a local file. Decode/intensity-window/channel decisions belong in that reader. Supply a stable `content_hash` for such resources so cache invalidation can detect content changes; the fallback ID/resource fingerprint cannot detect changed database contents.

## Splits

- `image`: class-stratified random four-role allocation; exact global role counts use largest remainders. Three sequential random states are `s`, `s+1`, `s+2`.
- `group`: whole-group, approximately class-balanced allocation using multiple greedy starts; patient/content group integrity takes precedence over exact ratios. All classes must appear in each internal role.
- `predefined`: use canonical `role` values `train`, `selection`, `calibration`, `evaluation`. Map arbitrary source names with source-level `role_map`.

One fixed predefined allocation uses one outer seed; duplicating the same role lists under several seed names would not create new random splits and is rejected. Separate imported allocations can be run as separate configurations and combined with their actual allocation identities.

The configured `internal_cohort` is split; other cohort names are external, with labels preserved for final evaluation. Only internal training images enter few-shot sampling. Missing classes or too few groups raise explicit errors. Four roles distinguish parameter updates, repeated epoch/configuration choice, calibration and final reporting. The ratios are configurable, not a statistically established universal optimum.

Cross-role group and content-hash overlaps are audited. The default policy is `error`; `report` retains them explicitly. This does not establish patient independence if no reliable patient IDs are supplied. Run `prepare` again after changing source tables/images; reuse a checkpoint only if its recorded cache/config/data membership matches.
