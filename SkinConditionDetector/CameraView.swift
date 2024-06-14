//
//  CameraView.swift
//  SkinConditionDetector
//
//  Created by Ananth Kashyap on 6/11/24.
//

import SwiftUI
import AVFoundation
import CoreML
import Vision

struct CameraView: View {
    @Binding var image: UIImage?
    @Binding var isShown: Bool
    @Binding var classifications: [(String, Float)]

    var body: some View {
        ZStack {
            CameraPreview(image: $image, isShown: $isShown, classifications: $classifications)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                Button(action: {
                    isShown = false
                }) {
                    Text("Cancel")
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.red)
                        .cornerRadius(10)
                }
                .padding(.bottom, 20)
            }
        }
    }
}

class CameraCoordinator: NSObject, AVCapturePhotoCaptureDelegate {
    var parent: CameraPreview
    
    init(parent: CameraPreview) {
        self.parent = parent
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation() else { return }
        if let uiImage = UIImage(data: data) {
            parent.image = uiImage
            parent.isShown = false
            parent.classifyImage(image: uiImage)
        }
    }
}

struct CameraPreview: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Binding var isShown: Bool
    @Binding var classifications: [(String, Float)]
    
    class CameraViewController: UIViewController {
        var captureSession: AVCaptureSession?
        var photoOutput: AVCapturePhotoOutput?
        var previewLayer: AVCaptureVideoPreviewLayer?
        var coordinator: CameraCoordinator?
        
        override func viewDidLoad() {
            super.viewDidLoad()
            
            captureSession = AVCaptureSession()
            guard let captureSession = captureSession else { return }
            
            guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
            let videoInput: AVCaptureDeviceInput
            
            do {
                videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
            } catch {
                return
            }
            
            if captureSession.canAddInput(videoInput) {
                captureSession.addInput(videoInput)
            } else {
                return
            }
            
            photoOutput = AVCapturePhotoOutput()
            if captureSession.canAddOutput(photoOutput!) {
                captureSession.addOutput(photoOutput!)
            } else {
                return
            }
            
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer?.frame = view.layer.bounds
            previewLayer?.videoGravity = .resizeAspectFill
            view.layer.addSublayer(previewLayer!)
            
            captureSession.startRunning()
            
            let captureButton = UIButton(type: .custom)
            captureButton.frame = CGRect(x: view.frame.midX - 35, y: view.frame.maxY - 100, width: 70, height: 70)
            captureButton.layer.cornerRadius = 35
            captureButton.backgroundColor = .red
            captureButton.addTarget(self, action: #selector(didTapCaptureButton), for: .touchUpInside)
            view.addSubview(captureButton)
        }
        
        @objc func didTapCaptureButton() {
            let settings = AVCapturePhotoSettings()
            photoOutput?.capturePhoto(with: settings, delegate: coordinator!)
        }
    }
    
    func makeCoordinator() -> CameraCoordinator {
        return CameraCoordinator(parent: self)
    }
    
    func makeUIViewController(context: Context) -> CameraViewController {
        let viewController = CameraViewController()
        viewController.coordinator = context.coordinator
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {}
    
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
