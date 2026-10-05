# Third-party components (Kotlin track)

Every dependency is open source under a permissive licence (no GPL/AGPL, no
closed-source SDKs such as Google ML Kit).

| Component | Version | Used by | Licence | Notes |
| --- | --- | --- | --- | --- |
| PP-OCRv5 latin mobile recognition model (PaddleOCR), ONNX export by RapidOCR | v5 | core (bundled in `models/`, APK assets) | Apache-2.0 | See `models/NOTICE`, `models/LICENSE-Apache-2.0.txt` |
| OpenCV (Java API) | 4.9.0 desktop / 4.12.0 Android | core, cli, app | Apache-2.0 | ArUco detection, homography warp, CLAHE, image I/O |
| `org.openpnp:opencv` (OpenCV packaged with desktop natives) | 4.9.0-0 | cli, tests | Apache-2.0 (OpenCV >= 4.5; the POM says "BSD License" but links OpenCV's own LICENSE) | |
| `org.opencv:opencv` (official Android AAR) | 4.12.0 | app | Apache-2.0 | |
| ONNX Runtime (`onnxruntime`, `onnxruntime-android`) | 1.30.0 | core, cli, app | MIT | |
| Kotlin standard library | 2.4.20 | all | Apache-2.0 | |
| kotlinx.serialization | 1.11.0 | core | Apache-2.0 | JSON in/out |
| Jetpack Compose (BOM 2026.06.01), AndroidX Activity, Lifecycle, Core | see `gradle/libs.versions.toml` | app | Apache-2.0 | |
| JUnit 5 / JUnit Platform | 6.1.3 | tests | EPL-2.0 | test-only, not distributed |
| Gradle wrapper | 9.8.0 | build | Apache-2.0 | `gradle/wrapper/` |

Data (not code, not committed here):

| Data | Licence | Notes |
| --- | --- | --- |
| `testdata/names/names_v1.json` (built from MTGJSON AtomicCards) | MIT (MTGJSON) | copied into the APK assets at build time |
| Card names | Magic: The Gathering is a trademark of Wizards of the Coast | unofficial fan content |
