"use client";

import {useState} from "react";

export default function Home() {
	const [image, setImage] = useState<File | null>(null);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [resultImage, setResultImage] = useState<string | null>(null); // NEW: Store drawn image
	const [results, setResults] = useState<any>(null);
	const [loading, setLoading] = useState(false);

	const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (file) {
			setImage(file);
			setPreviewUrl(URL.createObjectURL(file));
			setResults(null);
			setResultImage(null); // Reset when new image chosen
		}
	};

	const handleUpload = async () => {
		if (!image) return;
		setLoading(true);

		const formData = new FormData();
		formData.append("image", image);

		try {
			const response = await fetch("http://127.0.0.1:5000/predict", {
				method: "POST",
				body: formData,
			});

			const data = await response.json();
			setResults(data);
			if (data.image_with_boxes) {
				setResultImage(data.image_with_boxes); // Save the drawn image
			}
		} catch (error) {
			console.error("Error:", error);
			alert("Failed to connect to the AI backend.");
		}

		setLoading(false);
	};

	return (
		<main className="min-h-screen bg-gray-50 p-8 text-gray-900 font-sans">
			<div className="max-w-4xl mx-auto space-y-8">
				<div className="text-center">
					<h1 className="text-4xl font-bold text-blue-900 mb-2">
						PCB Defect Detection AI
					</h1>
					<p className="text-gray-600">
						Final Year Project - Automated Optical Inspection
					</p>
				</div>

				<div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 flex flex-col items-center">
					<input
						type="file"
						accept="image/*"
						onChange={handleImageChange}
						className="mb-4 block w-full max-w-sm text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
					/>
					<button
						onClick={handleUpload}
						disabled={!image || loading}
						className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
					>
						{loading ? "Analyzing Board..." : "Run AI Inspection"}
					</button>
				</div>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
					{/* CHANGED: Show the drawn image if it exists, otherwise show raw upload */}
					<div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 min-h-75 flex items-center justify-center">
						{resultImage ? (
							<img
								src={`data:image/jpeg;base64,${resultImage}`}
								alt="AI Result"
								className="max-w-full h-auto rounded-lg"
							/>
						) : previewUrl ? (
							<img
								src={previewUrl}
								alt="PCB Preview"
								className="max-w-full h-auto rounded-lg"
							/>
						) : (
							<p className="text-gray-400">No image selected</p>
						)}
					</div>

					<div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
						<h2 className="text-xl font-bold mb-4 border-b pb-2">
							Inspection Report
						</h2>
						{!results ? (
							<p className="text-gray-400 italic">
								Upload an image and run inspection to see results here.
							</p>
						) : (
							<div>
								<div
									className={`text-lg font-bold p-3 rounded-lg mb-4 ${results.defect_count > 0 ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800"}`}
								>
									{results.defect_count > 0
										? `❌ ${results.defect_count} Defect(s) Detected`
										: "✅ Board Passed Inspection (No Defects)"}
								</div>
								{results.defects.length > 0 && (
									<ul className="space-y-3">
										{results.defects.map((defect: any, index: number) => (
											<li
												key={index}
												className="bg-gray-50 p-3 rounded border border-gray-200"
											>
												<span className="font-semibold text-red-600 capitalize">
													{defect.class_name.replace("_", " ")}
												</span>
												<span className="text-sm text-gray-500 ml-2">
													(Confidence: {(defect.confidence * 100).toFixed(1)}%)
												</span>
											</li>
										))}
									</ul>
								)}
							</div>
						)}
					</div>
				</div>
			</div>
		</main>
	);
}
