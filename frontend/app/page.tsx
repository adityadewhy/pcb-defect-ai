"use client";

import {useState, useEffect, useRef} from "react";
import {
	PieChart,
	Pie,
	Cell,
	Tooltip,
	Legend,
	ResponsiveContainer,
} from "recharts";
import * as THREE from "three";

// Monochromatic color palette for Pie Chart
const MONO_COLORS = [
	"#000000",
	"#3f3f46",
	"#71717a",
	"#a1a1aa",
	"#d4d4d8",
	"#f4f4f5",
];

interface Defect {
	class_name: string;
	confidence: number;
}

interface PredictionResult {
	success: boolean;
	image_url?: string;
	defects?: Defect[];
}

export default function Home() {
	const [confidence, setConfidence] = useState<number>(0.04);
	const [selectedFile, setSelectedFile] = useState<File | null>(null);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [result, setResult] = useState<PredictionResult | null>(null);
	const [loading, setLoading] = useState<boolean>(false);

	// Three.js Canvas Reference
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	// 1. Initialize Three.js 3D Interactive Background
	useEffect(() => {
		if (!canvasRef.current) return;

		const canvas = canvasRef.current;
		const scene = new THREE.Scene();
		const camera = new THREE.PerspectiveCamera(
			60,
			window.innerWidth / window.innerHeight,
			0.1,
			1000,
		);
		camera.position.z = 15;

		const renderer = new THREE.WebGLRenderer({
			canvas,
			alpha: true,
			antialias: true,
		});
		renderer.setSize(window.innerWidth, window.innerHeight);
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

		// Create 3D Wireframe Grid / PCB Mesh
		const geometry = new THREE.PlaneGeometry(30, 30, 24, 24);
		const wireframe = new THREE.WireframeGeometry(geometry);
		const material = new THREE.LineBasicMaterial({
			color: 0x333333,
			transparent: true,
			opacity: 0.6,
		});
		const gridMesh = new THREE.LineSegments(wireframe, material);
		gridMesh.rotation.x = -Math.PI / 3;
		scene.add(gridMesh);

		// Floating Particles
		const particlesCount = 120;
		const posArray = new Float32Array(particlesCount * 3);

		for (let i = 0; i < particlesCount * 3; i++) {
			posArray[i] = (Math.random() - 0.5) * 40;
		}

		const particlesGeometry = new THREE.BufferGeometry();
		particlesGeometry.setAttribute(
			"position",
			new THREE.BufferAttribute(posArray, 3),
		);

		const particlesMaterial = new THREE.PointsMaterial({
			size: 0.08,
			color: 0xffffff,
			transparent: true,
			opacity: 0.4,
		});

		const particlesMesh = new THREE.Points(
			particlesGeometry,
			particlesMaterial,
		);
		scene.add(particlesMesh);

		// Mouse Tracking for Parallax Effect
		let mouseX = 0;
		let mouseY = 0;

		const handleMouseMove = (event: MouseEvent) => {
			mouseX = (event.clientX / window.innerWidth - 0.5) * 2;
			mouseY = (event.clientY / window.innerHeight - 0.5) * 2;
		};

		// Scroll Tracking for Rotation
		let scrollY = 0;
		const handleScroll = () => {
			scrollY = window.scrollY;
		};

		window.addEventListener("mousemove", handleMouseMove);
		window.addEventListener("scroll", handleScroll);

		// Resize Handler
		const handleResize = () => {
			camera.aspect = window.innerWidth / window.innerHeight;
			camera.updateProjectionMatrix();
			renderer.setSize(window.innerWidth, window.innerHeight);
		};
		window.addEventListener("resize", handleResize);

		// Animation Loop
		let animationFrameId: number;
		const animate = () => {
			animationFrameId = requestAnimationFrame(animate);

			// Smooth 3D Rotations
			gridMesh.rotation.z += 0.001;
			gridMesh.rotation.x = -Math.PI / 3 + mouseY * 0.1 + scrollY * 0.0005;
			gridMesh.rotation.y = mouseX * 0.15;

			particlesMesh.rotation.y += 0.0005;
			particlesMesh.position.y = -scrollY * 0.002;

			renderer.render(scene, camera);
		};

		animate();

		// Clean up
		return () => {
			window.removeEventListener("mousemove", handleMouseMove);
			window.removeEventListener("scroll", handleScroll);
			window.removeEventListener("resize", handleResize);
			cancelAnimationFrame(animationFrameId);
			geometry.dispose();
			material.dispose();
			particlesGeometry.dispose();
			particlesMaterial.dispose();
			renderer.dispose();
		};
	}, []);

	// 2. Handle File Selection
	const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files && e.target.files[0]) {
			const file = e.target.files[0];
			setSelectedFile(file);
			setPreviewUrl(URL.createObjectURL(file));
			setResult(null);
		}
	};

	// 3. Upload to Backend API
	const handleUpload = async () => {
		if (!selectedFile) return;
		setLoading(true);

		const formData = new FormData();
		formData.append("file", selectedFile);
		formData.append("confidence", confidence.toString());

		try {
			const response = await fetch("https://pcb-defect-ai-backend.onrender.com", {
				method: "POST",
				body: formData,
			});
			const data = await response.json();
			setResult(data);
		} catch (err) {
			console.error("Prediction failed:", err);
			alert(
				"Failed to connect to backend server. Make sure Flask app.py is running.",
			);
		} finally {
			setLoading(false);
		}
	};

	// 4. Export Multi-Page PDF Handler
	const handleDownloadPDF = async () => {
		const {toPng} = await import("html-to-image");
		const {default: jsPDF} = await import("jspdf");

		const page1El = document.getElementById("report-page-1");
		const page2El = document.getElementById("report-page-2");

		if (!page1El || !page2El) return;

		try {
			const pdf = new jsPDF("p", "mm", "a4");
			const pdfWidth = pdf.internal.pageSize.getWidth();
			const pdfHeight = pdf.internal.pageSize.getHeight();

			// Page 1 Render
			const imgData1 = await toPng(page1El, {quality: 0.98, cacheBust: true});
			const imgHeight1 =
				(page1El.offsetHeight * pdfWidth) / page1El.offsetWidth;
			pdf.addImage(
				imgData1,
				"PNG",
				0,
				0,
				pdfWidth,
				Math.min(imgHeight1, pdfHeight),
			);

			// Page 2 Render
			pdf.addPage();
			const imgData2 = await toPng(page2El, {quality: 0.98, cacheBust: true});
			const imgHeight2 =
				(page2El.offsetHeight * pdfWidth) / page2El.offsetWidth;
			pdf.addImage(
				imgData2,
				"PNG",
				0,
				0,
				pdfWidth,
				Math.min(imgHeight2, pdfHeight),
			);

			pdf.save("PCB_Defect_Inspection_Report.pdf");
		} catch (err) {
			console.error("PDF Generation Error:", err);
		}
	};

	// 5. Aggregate defect counts for Pie Chart
	const defectCounts =
		result?.defects?.reduce((acc: Record<string, number>, curr) => {
			acc[curr.class_name] = (acc[curr.class_name] || 0) + 1;
			return acc;
		}, {}) || {};

	const chartData = Object.keys(defectCounts).map((name) => ({
		name,
		value: defectCounts[name],
	}));

	return (
		<div className="relative min-h-screen bg-black text-white bg-grid-pattern overflow-x-hidden selection:bg-white selection:text-black">
			{/* 3D Three.js Interactive Background Canvas */}
			<canvas
				ref={canvasRef}
				className="fixed inset-0 pointer-events-none z-0"
			/>

			{/* Main Content Container */}
			<div className="relative z-10 py-12 px-4 max-w-4xl mx-auto space-y-12">
				{/* Header Section */}
				<header className="text-center space-y-4 animate-fade-in">
					<div className="inline-block border border-neutral-800 bg-neutral-950/80 px-4 py-1 rounded-full">
						<p className="text-xs font-mono uppercase tracking-widest text-neutral-400">
							[ SYSTEM V1.0 &bull; YOLOv8 VISION ]
						</p>
					</div>
					<h1 className="text-4xl md:text-6xl font-black tracking-tighter uppercase font-mono">
						PCB DEFECT DETECTOR
					</h1>
					<p className="text-neutral-400 text-sm md:text-base max-w-xl mx-auto font-light tracking-wide">
						Automated computer vision surface inspection and monochromatic
						diagnostic reporting.
					</p>
				</header>

				{/* Control Panel Card */}
				<section className="mono-card p-6 md:p-8 rounded-none border border-neutral-800 space-y-6 animate-fade-in">
					<div className="flex justify-between items-center border-b border-neutral-800 pb-3">
						<span className="text-xs font-mono uppercase tracking-widest text-neutral-400">
							01 / INPUT CONTROL
						</span>
						<span className="text-xs font-mono text-neutral-500">
							FORMAT: PNG, JPG, WEBP
						</span>
					</div>

					<div>
						<label className="block text-xs font-mono uppercase tracking-wider mb-2 text-neutral-300">
							SELECT PCB IMAGE FILE
						</label>
						<div className="relative flex items-center justify-center border border-dashed border-neutral-700 hover:border-white p-6 transition-all bg-neutral-950/50 cursor-pointer group">
							<input
								type="file"
								accept="image/*"
								onChange={handleFileChange}
								className="absolute inset-0 opacity-0 w-full h-full cursor-pointer z-10"
							/>
							<div className="text-center space-y-2">
								<div className="w-8 h-8 mx-auto border border-white/20 flex items-center justify-center font-mono text-xs group-hover:border-white transition-colors">
									+
								</div>
								<p className="text-sm font-mono tracking-tight text-neutral-300">
									{selectedFile
										? selectedFile.name
										: "DRAG & DROP OR CLICK TO UPLOAD"}
								</p>
							</div>
						</div>
					</div>

					{/* Confidence Threshold Slider */}
					<div className="space-y-3">
						<div className="flex justify-between text-xs font-mono uppercase tracking-wider">
							<span className="text-neutral-300">CONFIDENCE THRESHOLD</span>
							<span className="text-white font-bold bg-neutral-800 px-2 py-0.5 border border-neutral-700">
								{(confidence * 100).toFixed(0)}%
							</span>
						</div>
						<input
							type="range"
							min="0.01"
							max="1.00"
							step="0.01"
							value={confidence}
							onChange={(e) => setConfidence(parseFloat(e.target.value))}
							className="w-full h-1 bg-neutral-800 appearance-none cursor-pointer accent-white"
						/>
						<div className="flex justify-between text-[10px] font-mono text-neutral-500 uppercase">
							<span>SENSITIVE (4%)</span>
							<span>STRICT (100%)</span>
						</div>
					</div>

					{/* Run Inspection Button */}
					<button
						onClick={handleUpload}
						disabled={!selectedFile || loading}
						className="w-full mono-btn py-4 px-6 text-sm font-mono cursor-pointer disabled:cursor-not-allowed"
					>
						{loading ? "ANALYZING SURFACE MATRIX..." : "RUN INSPECTION LOGIC"}
					</button>
				</section>

				{/* Laser Scanner Preview Overlay */}
				{previewUrl && !result && loading && (
					<section className="mono-card p-6 border border-neutral-800 text-center space-y-4 animate-fade-in mono-scanner">
						<p className="text-xs font-mono uppercase tracking-widest text-neutral-400">
							SCANNING IN PROGRESS
						</p>
						<div className="flex justify-center">
							<img
								src={previewUrl}
								alt="Scanning"
								className="max-h-80 border border-neutral-800 object-contain grayscale"
							/>
						</div>
					</section>
				)}

				{/* Inspection Results Section */}
				{result && (
					<section className="space-y-8 animate-fade-in">
						{/* Export Toolbar */}
						<div className="flex justify-between items-center bg-neutral-950 p-4 border border-neutral-800">
							<div className="space-y-0.5 font-mono">
								<p className="text-xs uppercase tracking-widest text-white">
									STATUS: COMPLETE
								</p>
								<p className="text-[11px] text-neutral-500">
									DETECTED {result.defects?.length || 0} ANOMALY OBJECTS
								</p>
							</div>
							<button
								onClick={handleDownloadPDF}
								className="mono-btn px-5 py-2.5 text-xs font-mono cursor-pointer"
							>
								DOWNLOAD PDF REPORT
							</button>
						</div>

						{/* PAGE 1: PCB Image (Exportable White Background Card) */}
						<div
							id="report-page-1"
							className="p-8 bg-white text-black space-y-6 shadow-2xl flex flex-col justify-between border border-black"
						>
							<div>
								<div className="border-b-2 border-black pb-4 mb-6 flex justify-between items-center font-mono">
									<div>
										<h2 className="text-2xl font-black uppercase tracking-tight">
											PCB VISUAL INSPECTION
										</h2>
										<p className="text-xs text-neutral-600 uppercase">
											AUTOMATED OPTICAL DIAGNOSTIC REPORT
										</p>
									</div>
									<span className="text-xs bg-black text-white px-3 py-1 font-bold">
										PAGE 01 / 02
									</span>
								</div>

								<div className="text-center mb-4 text-xs font-mono uppercase tracking-widest text-neutral-500">
									BOUNDING BOX OVERLAY OVERVIEW
								</div>

								{/* PCB Image */}
								{(result.image_url || previewUrl) && (
									<div className="flex justify-center items-center my-4">
										<img
											src={result.image_url || previewUrl || ""}
											alt="Inspected PCB"
											className="max-h-115 w-auto border-2 border-black object-contain"
										/>
									</div>
								)}
							</div>

							<div className="text-[10px] font-mono text-center text-neutral-500 border-t border-neutral-300 pt-4 uppercase">
								AI PCB INSPECTOR SYSTEM &bull; CONFIDENTIAL AUDIT DATA
							</div>
						</div>

						{/* PAGE 2: Defect Analytics & Table */}
						<div
							id="report-page-2"
							className="p-8 bg-white text-black space-y-6 shadow-2xl flex flex-col justify-between border border-black"
						>
							<div>
								<div className="border-b-2 border-black pb-4 mb-6 flex justify-between items-center font-mono">
									<div>
										<h2 className="text-2xl font-black uppercase tracking-tight">
											DEFECT ANALYTICS
										</h2>
										<p className="text-xs text-neutral-600 uppercase">
											QUANTITATIVE SUMMARY BREAKDOWN
										</p>
									</div>
									<span className="text-xs bg-black text-white px-3 py-1 font-bold">
										PAGE 02 / 02
									</span>
								</div>

								{/* Monochrome Pie Chart */}
								{chartData.length > 0 ? (
									<div className="h-64 w-full mb-8 font-mono">
										<h3 className="text-xs font-bold mb-3 text-center uppercase tracking-widest text-neutral-700">
											DEFECT CLASS DISTRIBUTION
										</h3>
										<ResponsiveContainer width="100%" height="100%">
											<PieChart>
												<Pie
													data={chartData}
													cx="50%"
													cy="50%"
													outerRadius={80}
													dataKey="value"
													stroke="#000000"
													strokeWidth={2}
													label={(entry: {name?: string; percent?: number}) =>
														`${entry.name ?? ""} ${((entry.percent ?? 0) * 100).toFixed(0)}%`
													}
												>
													{chartData.map((_, index) => (
														<Cell
															key={`cell-${index}`}
															fill={MONO_COLORS[index % MONO_COLORS.length]}
														/>
													))}
												</Pie>
												<Tooltip
													contentStyle={{
														backgroundColor: "#000000",
														color: "#ffffff",
														border: "1px solid #ffffff",
													}}
												/>
												<Legend />
											</PieChart>
										</ResponsiveContainer>
									</div>
								) : (
									<p className="text-center font-mono text-xs uppercase text-neutral-500 py-6">
										NO ANOMALIES DETECTED FOR GRAPH ANALYSIS.
									</p>
								)}

								{/* Defect Table */}
								<div>
									<h3 className="text-xs font-mono font-bold mb-3 uppercase tracking-widest text-neutral-700">
										ANOMALY LOG MATRIX
									</h3>
									{result.defects && result.defects.length > 0 ? (
										<table className="w-full text-left border-collapse font-mono text-xs border-2 border-black">
											<thead>
												<tr className="bg-black text-white uppercase">
													<th className="p-3 border-r border-neutral-800">#</th>
													<th className="p-3 border-r border-neutral-800">
														DEFECT CLASS
													</th>
													<th className="p-3">CONFIDENCE</th>
												</tr>
											</thead>
											<tbody>
												{result.defects.map((defect, idx) => (
													<tr
														key={idx}
														className={
															idx % 2 === 0 ? "bg-neutral-100" : "bg-white"
														}
													>
														<td className="p-3 border-t border-r border-black font-bold">
															{idx + 1}
														</td>
														<td className="p-3 border-t border-r border-black font-bold uppercase">
															{defect.class_name}
														</td>
														<td className="p-3 border-t border-black font-mono">
															{(defect.confidence * 100).toFixed(1)}%
														</td>
													</tr>
												))}
											</tbody>
										</table>
									) : (
										<p className="text-black font-mono text-xs font-bold text-center py-4 bg-neutral-100 border-2 border-black uppercase">
											[ PASSED ] NO DEFECTS FOUND ABOVE THRESHOLD
										</p>
									)}
								</div>
							</div>

							<div className="text-[10px] font-mono text-center text-neutral-500 border-t border-neutral-300 pt-4 uppercase">
								AI PCB INSPECTOR SYSTEM &bull; END OF REPORT
							</div>
						</div>
					</section>
				)}

				{/* Credit / Resources Footer Section */}
				<footer className="border-t border-neutral-800 pt-8 mt-12 text-center font-mono space-y-4">
					<p className="text-xs uppercase tracking-widest text-neutral-400">
						[ CREDITS &amp; RESOURCES ]
					</p>
					<div className="flex flex-wrap justify-center gap-6 text-xs">
						<a
							href="https://colab.research.google.com/drive/1LdNzxeBYR8sazatWWknFP-2BDYkRcbvr?usp=sharing"
							target="_blank"
							rel="noopener noreferrer"
							className="text-neutral-300 hover:text-white underline underline-offset-4 cursor-pointer transition-colors"
						>
							Model Training &amp; Weights
						</a>
						<a
							href="https://github.com/adityadewhy/pcb-defect-ai"
							target="_blank"
							rel="noopener noreferrer"
							className="text-neutral-300 hover:text-white underline underline-offset-4 cursor-pointer transition-colors"
						>
							Code
						</a>
						<a
							href="https://universe.roboflow.com/x-2v7kv/pku-pcb-vpdup/"
							target="_blank"
							rel="noopener noreferrer"
							className="text-neutral-300 hover:text-white underline underline-offset-4 cursor-pointer transition-colors"
						>
							Dataset
						</a>
					</div>
				</footer>
			</div>
		</div>
	);
}
