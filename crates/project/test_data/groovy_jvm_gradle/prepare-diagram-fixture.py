#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZipInfo

parser = argparse.ArgumentParser(description='Create the offline Groovy dependency diagram fixture')
parser.add_argument('workspace', type=Path, help='existing directory for a new graph-project')
parser.add_argument('--ordered-tasks', action='store_true', help='include the ordered Gradle Task probes')
args = parser.parse_args()
workspace = args.workspace.resolve()
root = workspace / 'graph-project'
root.mkdir()
source = Path(__file__).resolve().parent
for wrapper_file in (
    'gradlew',
    'gradlew.bat',
    'gradle/wrapper/gradle-wrapper.jar',
    'gradle/wrapper/gradle-wrapper.properties',
):
    destination = root / wrapper_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / wrapper_file, destination)

for module in ('application', 'library'):
    (root / module).mkdir()
    (root / module / 'build.gradle').write_text("plugins { id 'java-library' }\n")
(root / 'settings.gradle').write_text("rootProject.name = 'groovy-graph-qualification'\ninclude 'application', 'library'\n")
probe = root / 'application/src/main/java/GraphProbe.java'
probe.parent.mkdir(parents=True)
probe.write_text('public final class GraphProbe {}\n')
(root / 'build.gradle').write_text("allprojects { group = 'verification'; version = '1.0' }\n")

members = {}


def package(name, dependencies=(), version='1.0'):
    members[name, version] = list(dependencies)
    artifact = f'{name}-{version}'
    path = root / 'graph-repository' / 'verification' / name / version
    path.mkdir(parents=True)
    header = '<project><modelVersion>4.0.0</modelVersion>'
    coordinate = f'<groupId>verification</groupId><artifactId>{name}</artifactId><version>{version}</version>'
    dependency_xml = ''.join(
        f'<dependency><groupId>verification</groupId><artifactId>{child}</artifactId><version>{requested}</version></dependency>'
        for child, requested in dependencies
    )
    (path / f'{artifact}.pom').write_text(f'{header}{coordinate}<dependencies>{dependency_xml}</dependencies></project>\n')
    with ZipFile(path / f'{artifact}.jar', 'w') as archive:
        archive.writestr(ZipInfo('META-INF/MANIFEST.MF'), 'Manifest-Version: 1.0\r\n\r\n')

for number in range(1, 13):
    package(f'common-{number:02d}')
package('shared-observer')
package('shared-version')
package('shared-version', version='2.0')
package('cycle-a', [('cycle-b', '1.0')])
package('cycle-b', [('cycle-a', '1.0')])
for branch in range(1, 13):
    leaves = []
    for leaf in range(1, 9):
        name = f'leaf-{branch:02d}-{leaf:02d}'
        first = (branch + leaf - 2) % 12 + 1
        second = (first + 4) % 12 + 1
        package(name, [
            (f'common-{first:02d}', '1.0'),
            (f'common-{second:02d}', '1.0'),
            ('shared-observer', '1.0'),
        ])
        leaves.append((name, '1.0'))
    package(f'branch-{branch:02d}', [*leaves, ('shared-version', '1.0' if branch < 7 else '2.0')])

direct = [(f'branch-{branch:02d}', '1.0') for branch in range(1, 13)] + [('cycle-a', '1.0'), ('missing', '1.0')]
declarations = ''.join(f"    qualificationGraph 'verification:{name}:{version}'\n" for name, version in direct)
with (root / 'application/build.gradle').open('a') as build_file:
    build_file.write("""
repositories { maven { url = uri(rootProject.file('graph-repository')) } }
configurations.create('qualificationGraph') { canBeResolved = true; canBeConsumed = false }
configurations.create('qualificationRuntime') {
    canBeResolved = true
    canBeConsumed = false
    extendsFrom configurations.qualificationGraph
}
dependencies {
    qualificationGraph project(':library')
""" + declarations + '}\n')
    if args.ordered_tasks:
        build_file.write("""
abstract class QualificationQueueWrite extends DefaultTask {
    private String message = 'default'
    private boolean markerEnabled = true

    @org.gradle.api.tasks.options.Option(option = 'project-dir', description = 'Literal fixture message')
    void setMessage(String value) { message = value }

    @org.gradle.api.tasks.options.Option(option = 'emit-marker', description = 'Record a primitive Boolean')
    void setEmitMarker(boolean value) { markerEnabled = value }

    @org.gradle.api.tasks.Internal
    @org.gradle.api.tasks.options.Option(option = 'record-receipt', description = 'Record a Gradle Boolean property')
    abstract org.gradle.api.provider.Property<Boolean> getRecordReceipt()

    @TaskAction
    void report() {
        def output = project.layout.buildDirectory.file('qualification/queue-message.txt').get().asFile
        output.parentFile.mkdirs()
        output.text = message
        new File(output.parentFile, 'queue-options.txt').text = "marker=${markerEnabled}, receipt=${recordReceipt.get()}"
        println 'QUALIFICATION_QUEUE_WRITE'
    }
}
tasks.register('qualificationQueueWrite', QualificationQueueWrite) {
    group = 'qualification'
    recordReceipt.convention(false)
}
def queueMessage = layout.buildDirectory.file('qualification/queue-message.txt')
def queueOptions = layout.buildDirectory.file('qualification/queue-options.txt')
tasks.register('qualificationQueueRead') {
    group = 'qualification'
    doLast {
        println 'QUALIFICATION_QUEUE_READ=' + queueMessage.get().asFile.text
        println 'QUALIFICATION_QUEUE_OPTIONS=' + queueOptions.get().asFile.text
    }
}
""")
identities = sorted(f'{name}:{version}' for (name, version) in members if (name, version) != ('shared-version', '1.0'))
assert len(identities) == 124
receipt = {
    'project': str(root), 'module': ':application', 'scope': 'qualificationGraph',
    'inherited_scope': 'qualificationRuntime',
    'resolved_coordinates': identities, 'projects': [':application', ':library'],
    'components': 126, 'edges': 413, 'shared_observer_root_paths': 96,
    'unresolved': 'verification:missing:1.0', 'conflict_selected': 'verification:shared-version:2.0',
    'files': {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob('*')) if path.is_file()}
}
(workspace / 'fixture-inputs.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(root)
