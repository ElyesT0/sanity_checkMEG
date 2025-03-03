// All URLs
const url_raw = [
  '#1',
  '#2',
  '#3',
  '#4',
  '#5',
  '#6',
  '#7',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_raw/sanity_check_sub08.html',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_raw/sanity_check_sub09.html',
  '#10',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_raw/sanity_check_sub11.html',
  '#12',
  '#13',
  '#14',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_raw/sanity_check_sub15.html',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_raw/sanity_check_sub16.html',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_raw/sanity_check_sub17.html',
];

const url_processed = [
  '#1',
  '#2',
  '#3',
  '#4',
  '#5',
  '#6',
  '#7',
  '#8',
  '#9',
  '#10',
  '#11',
  '#12',
  '#13',
  '#14',
  '#15',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_processed/post_sub16.html',
  'https://etabbane.fr/reports/meg/replaySeq/sanity_checks_processed/post_sub17.html',
];

// Data arrays for links
const rawReports = [];
const preProcessedReports = [];

for (let i = 1; i < url_raw.length; i++) {
  rawReports.push({ name: `Raw Report sub-${i + 1}`, url: url_raw[i] });
}

for (let i = 1; i < url_processed.length; i++) {
  preProcessedReports.push({
    name: `Pre-processed Report sub-${i + 1}`,
    url: url_processed[i],
  });
}

// Function to load links dynamically
function loadLinks(sectionId, links) {
  const container = document.getElementById(sectionId);
  links.forEach((report) => {
    if (!report.url.includes('#')) {
      const link = document.createElement('a');
      link.href = report.url;
      link.textContent = report.name;
      container.appendChild(link);
    }
  });
}

// Load links on page load
document.addEventListener('DOMContentLoaded', () => {
  loadLinks('raw-links', rawReports);
  loadLinks('pre-processed-links', preProcessedReports);
});
