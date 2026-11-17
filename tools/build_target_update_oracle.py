"""AGPL-3.0-only. Extract pinned target-update source; never format extracted code."""
import hashlib
import json
from pathlib import Path

REFERENCE = Path('C:/Users/a/RustroverProjects/GlobalSim')
manifest = json.loads(Path('docs/ORACLE_PROVENANCE.json').read_text())
for name in ('src/retail_lane.rs', 'src/overtaking.rs'):
    entry = next(e for e in manifest['files'] if e['origin'] == 'rust_reference' and e['path'] == name)
    assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == entry['sha256'], name
source = (REFERENCE / 'src/retail_lane.rs').read_text()

def cut(start, end):
    return source[source.index(start):source.index(end)]

horse = cut('#[derive(Clone, Copy, Debug)]\npub struct Horse', '#[derive(Clone, Copy, Debug)]\npub struct TargetParameters')
parameters = cut('#[derive(Clone, Copy, Debug)]\npub struct TargetParameters', 'impl TargetParameters')
context = cut('pub struct TargetContext', 'impl TargetState')
collect = cut('    pub fn collect(', '    /// `sorted` optionally')
cached = cut('    fn collect_cached', '    pub fn update_targets')
update = cut('    pub fn update_targets', '    pub fn target(')
front = cut('pub(crate) fn front_index_iter', 'pub fn side_room')
header = '// AGPL-3.0-only. Generated original-source subset; see TARGET_UPDATE_ORACLE.md.\n'
header += '#[path="lane_query_oracle.rs"] mod query;\nuse query::*;\n'
header += 'mod overtaking {\n' + (REFERENCE / 'src/overtaking.rs').read_text() + '\n}\n'
output = header + horse + parameters + context + 'impl TargetState {\n'
output += collect + cached + update + '}\n' + front
output += Path('tools/target_update_harness.rs').read_text()
Path('tools/target_update_oracle.rs').write_text(output, encoding='utf-8')
