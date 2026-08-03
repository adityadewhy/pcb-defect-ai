'use client';

import { useState } from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899'];

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
  const [confidence, setConfidence] = useState<number>(0.25);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

 
  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('confidence', confidence.toString());

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Failed to connect to backend server. Make sure Flask app.py is running.');
    } finally {
      setLoading(false);
    }
  };


  const handleDownloadPDF = async () => {
    const { toPng } = await import('html-to-image');
    const { default: jsPDF } = await import('jspdf');

    const element = document.getElementById('inspection-report');
    if (!element) return;

    try {
      const dataUrl = await toPng(element, { quality: 0.98, cacheBust: true });

      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (element.offsetHeight * pdfWidth) / element.offsetWidth;

      pdf.addImage(dataUrl, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('PCB_Defect_Inspection_Report.pdf');
    } catch (err) {
      console.error('PDF Generation Error:', err);
    }
  };
  
  const defectCounts = result?.defects?.reduce((acc: Record<string, number>, curr) => {
    acc[curr.class_name] = (acc[curr.class_name] || 0) + 1;
    return acc;
  }, {}) || {};

  const chartData = Object.keys(defectCounts).map((name) => ({
    name,
    value: defectCounts[name],
  }));

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6 text-white min-h-screen bg-slate-900">
      <h1 className="text-3xl font-bold text-center mb-6">AI PCB Defect Inspector</h1>


      <div className="bg-slate-800 p-6 rounded-xl shadow-lg space-y-4">
        <div>
          <label className="block text-sm font-semibold mb-2">Select PCB Image:</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
          />
        </div>

   
        <div>
          <label className="block text-sm font-semibold mb-2">
            AI Detection Threshold: <span className="text-blue-400">{(confidence * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range"
            min="0.10"
            max="1.00"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>10% (Sensitive)</span>
            <span>100% (Strict)</span>
          </div>
        </div>

        <button
          onClick={handleUpload}
          disabled={!selectedFile || loading}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white font-bold py-2 rounded-lg transition"
        >
          {loading ? 'Analyzing Image...' : 'Run PCB Inspection'}
        </button>
      </div>

    
      {result && (
        <div className="space-y-4">
          <div className="flex justify-end">
            <button
              onClick={handleDownloadPDF}
              className="bg-green-600 hover:bg-green-700 text-white font-semibold px-4 py-2 rounded-lg shadow"
            >
              📥 Download PDF Report
            </button>
          </div>

      
          <div id="inspection-report" className="p-6 bg-white text-slate-900 rounded-xl space-y-6">
            <h2 className="text-2xl font-bold border-b pb-2">Inspection Report</h2>

          
            {(result.image_url || previewUrl) && (
              <div className="flex justify-center">
                <img
                  src={result.image_url || previewUrl || ''}
                  alt="Inspected PCB"
                  className="max-h-80 rounded shadow border"
                />
              </div>
            )}

         
            {chartData.length > 0 && (
              <div className="h-64 w-full">
                <h3 className="text-lg font-semibold mb-2 text-center">Defect Breakdown</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={(entry: { name?: string; percent?: number }) =>
                        `${entry.name ?? ''} ${((entry.percent ?? 0) * 100).toFixed(0)}%`
                      }
                    >
                      {chartData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

         
            <div>
              <h3 className="text-lg font-semibold mb-2">Detected Defects List</h3>
              {result.defects && result.defects.length > 0 ? (
                <table className="w-full text-left border-collapse border text-sm">
                  <thead>
                    <tr className="bg-slate-100 border-b">
                      <th className="p-2 border">#</th>
                      <th className="p-2 border">Defect Type</th>
                      <th className="p-2 border">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.defects.map((defect, idx) => (
                      <tr key={idx} className="border-b">
                        <td className="p-2 border">{idx + 1}</td>
                        <td className="p-2 border">{defect.class_name}</td>
                        <td className="p-2 border">{(defect.confidence * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-green-600 font-medium">No defects detected above threshold.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}