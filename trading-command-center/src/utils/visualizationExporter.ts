import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

interface ExportOptions {
  format: 'png' | 'pdf' | 'svg' | 'csv';
  filename?: string;
  quality?: number;
  width?: number;
  height?: number;
  includeData?: boolean;
  title?: string;
  subtitle?: string;
  metadata?: { [key: string]: any };
}

interface ChartData {
  labels?: string[];
  datasets?: Array<{
    label?: string;
    data: any[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
  }>;
  [key: string]: any;
}

export class VisualizationExporter {
  private static instance: VisualizationExporter;

  public static getInstance(): VisualizationExporter {
    if (!VisualizationExporter.instance) {
      VisualizationExporter.instance = new VisualizationExporter();
    }
    return VisualizationExporter.instance;
  }

  /**
   * Export chart container to image/PDF
   */
  async exportChart(
    element: HTMLElement, 
    options: ExportOptions
  ): Promise<void> {
    const { format, filename, quality = 1.0 } = options;
    
    try {
      if (format === 'pdf') {
        await this.exportToPDF(element, options);
      } else if (format === 'png') {
        await this.exportToPNG(element, options);
      } else if (format === 'svg') {
        await this.exportToSVG(element, options);
      }
    } catch (error) {
      console.error('Export failed:', error);
      throw new Error(`Failed to export chart: ${error}`);
    }
  }

  /**
   * Export chart data to CSV
   */
  exportDataToCSV(
    data: ChartData | ChartData[], 
    filename: string
  ): void {
    const dataArray = Array.isArray(data) ? data : [data];
    
    if (dataArray.length === 0) {
      throw new Error('No data to export');
    }

    // Convert data to CSV format
    const csvContent = this.convertToCSV(dataArray);
    
    // Create and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `${filename}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  /**
   * Export multiple charts to a single PDF report
   */
  async exportReport(
    elements: Array<{ element: HTMLElement; title: string; description?: string }>,
    options: ExportOptions
  ): Promise<void> {
    const { filename = 'trading-report', title = 'Trading Analysis Report' } = options;
    
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });

    // Add title page
    this.addTitlePage(pdf, title);
    
    let yPosition = 40;
    
    for (let i = 0; i < elements.length; i++) {
      const { element, title: chartTitle, description } = elements[i];
      
      if (i > 0) {
        pdf.addPage();
      }
      
      // Add chart title
      pdf.setFontSize(16);
      pdf.setTextColor(0, 255, 0); // Matrix green
      pdf.text(chartTitle, 20, yPosition);
      
      if (description) {
        pdf.setFontSize(10);
        pdf.setTextColor(100, 100, 100);
        const lines = pdf.splitTextToSize(description, 170);
        pdf.text(lines, 20, yPosition + 10);
        yPosition += 20 + (lines.length * 5);
      }
      
      // Convert element to image
      const canvas = await html2canvas(element, {
        backgroundColor: '#0a0a0a',
        scale: 2,
        useCORS: true,
        allowTaint: true
      });
      
      const imgData = canvas.toDataURL('image/png');
      const imgWidth = 170; // PDF width in mm
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      
      // Add image to PDF
      pdf.addImage(imgData, 'PNG', 20, yPosition + 10, imgWidth, imgHeight);
      
      // Add timestamp
      pdf.setFontSize(8);
      pdf.setTextColor(100, 100, 100);
      const timestamp = new Date().toLocaleString();
      pdf.text(`Generated: ${timestamp}`, 20, yPosition + imgHeight + 20);
      
      yPosition = 0; // Reset for next page
    }
    
    // Save the PDF
    pdf.save(`${filename}.pdf`);
  }

  /**
   * Export chart as PNG
   */
  private async exportToPNG(
    element: HTMLElement, 
    options: ExportOptions
  ): Promise<void> {
    const { filename = 'chart', width = 1920, height = 1080, quality = 1.0 } = options;
    
    const canvas = await html2canvas(element, {
      backgroundColor: '#0a0a0a',
      scale: quality,
      width,
      height,
      useCORS: true,
      allowTaint: true
    });
    
    // Create download link
    canvas.toBlob((blob) => {
      if (blob) {
        const link = document.createElement('a');
        link.download = `${filename}.png`;
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
      }
    }, 'image/png', quality);
  }

  /**
   * Export chart as SVG
   */
  private async exportToSVG(
    element: HTMLElement, 
    options: ExportOptions
  ): Promise<void> {
    const { filename = 'chart' } = options;
    
    // Find SVG elements within the container
    const svgElements = element.querySelectorAll('svg');
    
    if (svgElements.length === 0) {
      throw new Error('No SVG elements found in the specified element');
    }
    
    // Export each SVG
    svgElements.forEach((svg, index) => {
      const svgData = new XMLSerializer().serializeToString(svg);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      
      const link = document.createElement('a');
      link.download = index === 0 ? `${filename}.svg` : `${filename}-${index + 1}.svg`;
      link.href = URL.createObjectURL(svgBlob);
      link.click();
      URL.revokeObjectURL(link.href);
    });
  }

  /**
   * Export chart to PDF
   */
  private async exportToPDF(
    element: HTMLElement, 
    options: ExportOptions
  ): Promise<void> {
    const { filename = 'chart', title, subtitle } = options;
    
    const canvas = await html2canvas(element, {
      backgroundColor: '#0a0a0a',
      scale: 2,
      useCORS: true,
      allowTaint: true
    });
    
    const pdf = new jsPDF({
      orientation: 'landscape',
      unit: 'mm',
      format: 'a4'
    });
    
    // Add title and subtitle if provided
    if (title) {
      pdf.setFontSize(20);
      pdf.setTextColor(0, 255, 0); // Matrix green
      pdf.text(title, 20, 20);
    }
    
    if (subtitle) {
      pdf.setFontSize(12);
      pdf.setTextColor(100, 100, 100);
      pdf.text(subtitle, 20, 30);
    }
    
    // Add the chart image
    const imgData = canvas.toDataURL('image/png');
    const pdfWidth = 280; // Landscape A4 width
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    
    pdf.addImage(imgData, 'PNG', 10, 45, pdfWidth, pdfHeight);
    
    // Add timestamp
    pdf.setFontSize(8);
    pdf.setTextColor(100, 100, 100);
    const timestamp = new Date().toLocaleString();
    pdf.text(`Generated: ${timestamp}`, 10, 200);
    
    // Save the PDF
    pdf.save(`${filename}.pdf`);
  }

  /**
   * Convert chart data to CSV format
   */
  private convertToCSV(dataArray: ChartData[]): string {
    const allKeys = new Set<string>();
    
    // Collect all possible keys
    dataArray.forEach(data => {
      if (data.labels) {
        data.labels.forEach(label => allKeys.add(label));
      }
      data.datasets?.forEach(dataset => {
        dataset.data.forEach((_, index) => {
          allKeys.add(index.toString());
        });
      });
    });
    
    const headers = Array.from(allKeys);
    const csvRows = [];
    
    // Add headers
    csvRows.push(['Metric', ...headers].join(','));
    
    // Add data rows
    dataArray.forEach(data => {
      if (data.datasets) {
        data.datasets.forEach((dataset, datasetIndex) => {
          const row = [
            dataset.label || `Dataset ${datasetIndex + 1}`,
            ...headers.map(header => {
              const index = parseInt(header);
              const value = dataset.data[index];
              return value !== undefined ? value.toString() : '';
            })
          ];
          csvRows.push(row.join(','));
        });
      }
    });
    
    return csvRows.join('\n');
  }

  /**
   * Add title page to PDF
   */
  private addTitlePage(pdf: jsPDF, title: string): void {
    pdf.setFontSize(24);
    pdf.setTextColor(0, 255, 0); // Matrix green
    pdf.text(title, 20, 50);
    
    pdf.setFontSize(12);
    pdf.setTextColor(100, 100, 100);
    pdf.text('Trading Analysis Report', 20, 70);
    
    // Add generation timestamp
    const timestamp = new Date().toLocaleString();
    pdf.text(`Generated: ${timestamp}`, 20, 90);
    
    // Add some decorative elements (simple lines)
    pdf.setDrawColor(0, 255, 0);
    pdf.line(20, 100, 190, 100);
  }

  /**
   * Copy chart to clipboard as image
   */
  async copyChartToClipboard(element: HTMLElement): Promise<void> {
    try {
      const canvas = await html2canvas(element, {
        backgroundColor: '#0a0a0a',
        scale: 2,
        useCORS: true,
        allowTaint: true
      });
      
      canvas.toBlob(async (blob) => {
        if (blob && navigator.clipboard && 'write' in navigator.clipboard) {
          try {
            const item = new ClipboardItem({ 'image/png': blob });
            await navigator.clipboard.write([item]);
          } catch (error) {
            console.error('Failed to copy to clipboard:', error);
            throw new Error('Failed to copy chart to clipboard');
          }
        }
      }, 'image/png');
    } catch (error) {
      console.error('Failed to copy chart:', error);
      throw new Error('Failed to copy chart to clipboard');
    }
  }

  /**
   * Generate shareable link for chart (base64 encoded)
   */
  generateShareableLink(element: HTMLElement): Promise<string> {
    return new Promise(async (resolve, reject) => {
      try {
        const canvas = await html2canvas(element, {
          backgroundColor: '#0a0a0a',
          scale: 1,
          useCORS: true,
          allowTaint: true
        });
        
        const dataUrl = canvas.toDataURL('image/png');
        const base64Data = dataUrl.split(',')[1];
        const shareableUrl = `data:image/png;base64,${base64Data}`;
        
        resolve(shareableUrl);
      } catch (error) {
        reject(error);
      }
    });
  }
}

// Export utility functions
export const exportUtils = {
  /**
   * Quick export function for common use cases
   */
  quickExport: async (
    element: HTMLElement, 
    format: 'png' | 'pdf' = 'png',
    filename?: string
  ): Promise<void> => {
    const exporter = VisualizationExporter.getInstance();
    await exporter.exportChart(element, {
      format,
      filename: filename || 'chart-export'
    });
  },

  /**
   * Export chart data as CSV
   */
  exportData: (data: ChartData | ChartData[], filename: string): void => {
    const exporter = VisualizationExporter.getInstance();
    exporter.exportDataToCSV(data, filename);
  },

  /**
   * Create export buttons for a chart container
   */
  createExportButtons: (
    container: HTMLElement,
    filename?: string,
    includeData?: boolean
  ): HTMLElement => {
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'export-buttons flex gap-2 mt-2';
    
    const buttons = [
      { label: 'Export PNG', format: 'png' as const },
      { label: 'Export PDF', format: 'pdf' as const },
      { label: 'Export SVG', format: 'svg' as const },
      { label: 'Copy Image', format: 'clipboard' as const },
    ];
    
    buttons.forEach(button => {
      const btn = document.createElement('button');
      btn.className = 'px-3 py-1 bg-matrix-green text-matrix-black text-sm font-mono rounded hover:bg-matrix-green/80 transition-colors';
      btn.textContent = button.label;
      
      btn.addEventListener('click', async () => {
        try {
          const exporter = VisualizationExporter.getInstance();
          
          if (button.format === 'clipboard') {
            await exporter.copyChartToClipboard(container);
          } else {
            await exporter.exportChart(container, {
              format: button.format,
              filename: filename || 'chart-export'
            });
          }
        } catch (error) {
          console.error('Export failed:', error);
          alert(`Export failed: ${error}`);
        }
      });
      
      buttonContainer.appendChild(btn);
    });
    
    return buttonContainer;
  }
};

export default VisualizationExporter;