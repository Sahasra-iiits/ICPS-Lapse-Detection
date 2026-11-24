`vision_bundle.mjs` and `wasm/vision_wasm_internal.{js,wasm}` are from the npm package
[`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) version 1.2.2
(Google, Apache License 2.0), unmodified. `../model/face_landmarker.task` is MediaPipe's Face
Landmarker model (Apache License 2.0).

Chrome extensions can't load code from the internet, so these files are bundled here.
