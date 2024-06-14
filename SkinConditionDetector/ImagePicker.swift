//
//  ImagePicker.swift
//  SkinConditionDetector
//
//  Created by Ananth Kashyap on 6/11/24.
//

import SwiftUI
import PhotosUI
import CoreML
import Vision

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var images: [UIImage]
    @Binding var classifications: [(String, Float)]

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration()
        configuration.filter = .images
        configuration.selectionLimit = 3  // Allow up to 3 images

        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        var parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.images.removeAll()
            parent.classifications.removeAll()

            for result in results {
                result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] (object, error) in
                    if let image = object as? UIImage {
                        DispatchQueue.main.async {
                            self?.parent.images.append(image)
                            self?.parent.classifyImage(image: image)
                        }
                    }
                }
            }
            picker.dismiss(animated: true)
        }
    }

    func classifyImage(image: UIImage) {
        let modelConfiguration = MLModelConfiguration()

        guard let modelURL = Bundle.main.url(forResource: "final_skin_condition_model", withExtension: "mlmodelc") else {
            classifications.append(("Model URL not found.", 0))
            return
        }

        do {
            let model = try final_skin_condition_model(contentsOf: modelURL, configuration: modelConfiguration)
            let vnModel = try VNCoreMLModel(for: model.model)

            let request = VNCoreMLRequest(model: vnModel) { request, error in
                if let error = error {
                    self.classifications.append(("Error: \(error.localizedDescription)", 0))
                    return
                }

                if let results = request.results as? [VNClassificationObservation], let firstResult = results.first {
                    self.classifications.append((firstResult.identifier, firstResult.confidence))
                } else {
                    self.classifications.append(("No results found", 0))
                }
            }

            guard let ciImage = CIImage(image: image) else {
                classifications.append(("Unable to convert UIImage to CIImage", 0))
                return
            }

            let handler = VNImageRequestHandler(ciImage: ciImage)
            try handler.perform([request])
        } catch {
            classifications.append(("Failed to load model: \(error.localizedDescription)", 0))
        }
    }
}
