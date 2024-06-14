//
//  ContentView.swift
//  SkinConditionDetector
//
//  Created by Ananth Kashyap on 6/10/24.
//
import SwiftUI
import PhotosUI
import CoreML
import Vision

struct ContentView: View {
    @State private var images: [UIImage] = []
    @State private var classifications: [(String, Float)] = []
    @State private var isImagePickerPresented: Bool = false
    @State private var isCameraPresented: Bool = false
    @State private var selectedImage: UIImage?

    func confidenceLabel(for confidence: Float) -> String {
        switch confidence {
        case 8...:
            return "High Confidence"
        case 6..<8:
            return "Medium Confidence"
        default:
            return "Low Confidence"
        }
    }

    var body: some View {
        NavigationView {
            VStack {
                Spacer(minLength: 30)

                Text("Skin Condition Classifier")
                    .font(.title) // Bigger font size
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity, alignment: .center)

                if images.isEmpty {
                    VStack {
                        Image(systemName: "photo")
                            .resizable()
                            .scaledToFit()
                            .frame(height: 150)
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .shadow(radius: 10)
                            .padding(.horizontal)
                            .opacity(0.3) // Make placeholder image less bold

                        VStack {
                            Text("Condition: N/A")
                                .font(.body)
                                .fontWeight(.bold)
                                .foregroundColor(.primary)
                                .padding(.horizontal)
                            Text("Confidence: N/A")
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                    }
                } else if images.count == 1 {
                    VStack {
                        Image(uiImage: images.first!)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 150)
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .shadow(radius: 10)
                            .padding(.horizontal)

                        if let index = images.firstIndex(of: images.first!), classifications.indices.contains(index) {
                            let (condition, confidence) = classifications[index]
                            VStack {
                                Text(condition)
                                    .font(.body)
                                    .fontWeight(.bold)
                                    .foregroundColor(.primary)
                                    .padding(.horizontal)
                                Text(confidenceLabel(for: confidence))
                                    .font(.body)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                } else {
                    ScrollView(.horizontal, showsIndicators: true) {
                        HStack(spacing: 10) {
                            ForEach(images, id: \.self) { image in
                                VStack {
                                    Image(uiImage: image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(height: 150)
                                        .clipShape(RoundedRectangle(cornerRadius: 20))
                                        .shadow(radius: 10)
                                        .padding(.horizontal)

                                    if let index = images.firstIndex(of: image), classifications.indices.contains(index) {
                                        let (condition, confidence) = classifications[index]
                                        VStack {
                                            Text(condition)
                                                .font(.body)
                                                .fontWeight(.bold)
                                                .foregroundColor(.primary)
                                                .padding(.horizontal)
                                            Text(confidenceLabel(for: confidence))
                                                .font(.body)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                }
                            }
                        }
                        .padding(.vertical)
                    }
                }

                if !classifications.isEmpty {
                    VStack(alignment: .center, spacing: 10) {
                        if classifications.allSatisfy({ $0.0 == classifications.first!.0 }) {
                            Text("All images classify as:")
                                .font(.headline)
                                .foregroundColor(.primary)
                                .frame(maxWidth: .infinity, alignment: .center)
                            Text(classifications.first!.0)
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(.primary)
                                .frame(maxWidth: .infinity, alignment: .center)
                            Text(confidenceLabel(for: classifications.first!.1))
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity, alignment: .center)
                        } else {
                            VStack(alignment: .leading, spacing: 10) {
                                ForEach(0..<classifications.count, id: \.self) { index in
                                    let (condition, confidence) = classifications[index]
                                    VStack(alignment: .leading) {
                                        Text("Image \(index + 1):")
                                            .font(.headline)
                                            .foregroundColor(.primary)
                                        Text(condition)
                                            .font(.title3)
                                            .fontWeight(.semibold)
                                            .foregroundColor(.primary)
                                        Text("(\(confidenceLabel(for: confidence)) Prediction)")
                                            .font(.subheadline)
                                            .foregroundColor(.primary)
                                    }
                                    .padding(.bottom, 10)
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 20)
                    .background(Color(UIColor.systemGray6))
                    .cornerRadius(10)
                    .padding(.horizontal)
                } else {
                    Text("No images selected")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                        .padding()
                }

                Spacer()

                HStack {
                    Button(action: {
                        isImagePickerPresented = true
                    }) {
                        Text("Select Images")
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            .padding(.horizontal)
                    }

                    Button(action: {
                        selectedImage = nil
                        isCameraPresented = true
                    }) {
                        Text("Take Photo")
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            .padding(.horizontal)
                    }
                }
            }
            .sheet(isPresented: $isImagePickerPresented) {
                ImagePicker(images: self.$images, classifications: self.$classifications)
            }
            .fullScreenCover(isPresented: $isCameraPresented) {
                CameraView(image: self.$selectedImage, isShown: self.$isCameraPresented, classifications: self.$classifications)
                    .onChange(of: selectedImage) { _, newImage in
                        if let newImage = newImage {
                            images.append(newImage)
                        }
                    }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: ContentView {
        ContentView()
    }
}
