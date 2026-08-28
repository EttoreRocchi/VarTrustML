"""
Shared CSS and JavaScript constants for HTML report generation.

The single definition of all HTML report styling, shared by
HTMLCompareReporter, HTMLTrainReporter, and HTMLCrossDatasetReporter.

Constants
---------
REPORT_CSS_BASE : str
    Base CSS styles shared by all report types.
CSS_SLATE_BLUE_THEME : str
    Slate blue color theme for compare-models reports.
CSS_TEAL_THEME : str
    Teal color theme for train reports.
CSS_TAUPE_THEME : str
    Warm taupe color theme for evaluate reports.
CSS_MAUVE_THEME : str
    Mauve color theme for cross-dataset reports.
REPORT_JS : str
    JavaScript for collapsible sections.
"""

# Base CSS shared by all report types (color-neutral)
REPORT_CSS_BASE = """
/* Base styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 30px 40px;
    background-color: #f8f9fa;
    color: #212529;
    line-height: 1.6;
    font-size: 16px;
}

/* Headings */
h1 {
    color: #1a1a1a;
    font-size: 2.5em;
    font-weight: 600;
    padding-bottom: 15px;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}

h2 {
    color: #2c3e50;
    font-size: 1.75em;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 20px;
    padding: 15px 20px;
    cursor: pointer;
    user-select: none;
    position: relative;
    border-radius: 4px;
    transition: all 0.2s ease;
}

h2:hover {
    transform: translateX(2px);
}

h2::after {
    content: '▼';
    position: absolute;
    right: 20px;
    font-size: 0.7em;
    transition: transform 0.3s ease;
}

h2.collapsed::after {
    transform: rotate(-90deg);
}

h3 {
    color: #495057;
    font-size: 1.3em;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #dee2e6;
}

h4 {
    color: #6c757d;
    font-size: 1.1em;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Paragraphs and text */
p {
    margin: 12px 0;
    line-height: 1.7;
}

em {
    color: #6c757d;
    font-style: italic;
}

.timestamp {
    color: #6c757d;
    font-size: 0.95em;
    margin-top: 5px;
    font-weight: 500;
}

ul {
    margin: 15px 0;
    padding-left: 25px;
}

li {
    margin: 8px 0;
    line-height: 1.6;
}

/* Section containers */
.section {
    background: white;
    padding: 30px;
    margin: 25px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
}

.section-content {
    overflow: hidden;
    transition: max-height 0.3s ease-out;
    padding-top: 10px;
}

.section-content.collapsed {
    max-height: 0 !important;
    padding: 0;
}

/* Tables */
.info-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 20px 0;
    font-size: 0.95em;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    overflow: hidden;
}

.info-table th {
    color: white;
    font-weight: 600;
    padding: 14px 16px;
    text-align: left;
    font-size: 0.95em;
    letter-spacing: 0.3px;
}

.info-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #e9ecef;
    color: #495057;
}

.info-table tr:last-child td {
    border-bottom: none;
}

.info-table tr:nth-child(even) {
    background-color: #f8f9fa;
}

.info-table tr:hover {
    transition: background-color 0.15s ease;
}

.info-table strong {
    font-weight: 600;
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 20px;
        font-size: 14px;
    }

    h1 {
        font-size: 2em;
    }

    h2 {
        font-size: 1.4em;
        padding: 12px 15px;
    }

    h3 {
        font-size: 1.15em;
    }

    .section {
        padding: 20px;
    }

    .info-table {
        font-size: 0.85em;
    }

    .info-table th,
    .info-table td {
        padding: 10px 12px;
    }
}
"""

# Slate Blue theme for compare-models reports
CSS_SLATE_BLUE_THEME = """
/* Slate Blue theme colors - compare-models */
h1 {
    border-bottom: 4px solid #5c6b8a;
}

h2 {
    border-left: 5px solid #5c6b8a;
    background: linear-gradient(to right, #eef1f5 0%, transparent 100%);
}

h2:hover {
    background: linear-gradient(to right, #e8ecf2 0%, transparent 100%);
}

h2::after {
    color: #5c6b8a;
}

.info-table th {
    background: linear-gradient(to bottom, #5c6b8a, #4a5a78);
    text-transform: uppercase;
}

.info-table tr:hover {
    background-color: #e8ecf2;
}

.info-table strong {
    color: #4a5a78;
}
"""

# Teal theme for train reports
CSS_TEAL_THEME = """
/* Teal theme colors - train */
h1 {
    border-bottom: 4px solid #5a8a7a;
}

h2 {
    border-left: 5px solid #5a8a7a;
    background: linear-gradient(to right, #eef5f3 0%, transparent 100%);
}

h2:hover {
    background: linear-gradient(to right, #e8f2ef 0%, transparent 100%);
}

h2::after {
    color: #5a8a7a;
}

.info-table th {
    background: linear-gradient(to bottom, #5a8a7a, #4a7a6a);
    text-transform: uppercase;
}

.info-table tr:hover {
    background-color: #e8f2ef;
}

.info-table strong {
    color: #4a7a6a;
}
"""

# Warm Taupe theme for evaluate reports
CSS_TAUPE_THEME = """
/* Warm Taupe theme colors - evaluate */
h1 {
    border-bottom: 4px solid #8a7a6a;
}

h2 {
    border-left: 5px solid #8a7a6a;
    background: linear-gradient(to right, #f5f3f0 0%, transparent 100%);
}

h2:hover {
    background: linear-gradient(to right, #f2efe8 0%, transparent 100%);
}

h2::after {
    color: #8a7a6a;
}

.info-table th {
    background: linear-gradient(to bottom, #8a7a6a, #7a6a5a);
    text-transform: uppercase;
}

.info-table tr:hover {
    background-color: #f2efe8;
}

.info-table strong {
    color: #7a6a5a;
}
"""

# Mauve theme for cross-dataset reports
CSS_MAUVE_THEME = """
/* Mauve theme colors - cross-dataset */
h1 {
    border-bottom: 4px solid #7a6a8a;
}

h2 {
    border-left: 5px solid #7a6a8a;
    background: linear-gradient(to right, #f3f0f5 0%, transparent 100%);
}

h2:hover {
    background: linear-gradient(to right, #efe8f2 0%, transparent 100%);
}

h2::after {
    color: #7a6a8a;
}

.info-table th {
    background: linear-gradient(to bottom, #7a6a8a, #6a5a7a);
    text-transform: uppercase;
}

.info-table tr:hover {
    background-color: #efe8f2;
}

.info-table strong {
    color: #6a5a7a;
}
"""

# JavaScript for collapsible sections
REPORT_JS = """
document.addEventListener('DOMContentLoaded', function() {
    // Make all sections expandable/collapsible
    const sections = document.querySelectorAll('.section');

    sections.forEach(function(section) {
        const h2 = section.querySelector('h2');
        if (h2) {
            // Wrap content after h2 in a div
            const content = document.createElement('div');
            content.className = 'section-content';

            let sibling = h2.nextElementSibling;
            while (sibling) {
                const next = sibling.nextElementSibling;
                content.appendChild(sibling);
                sibling = next;
            }

            section.appendChild(content);

            // Check if section should be initially collapsed
            const initiallyCollapsed = section.getAttribute('data-initially-collapsed') === 'true';

            if (initiallyCollapsed) {
                // Start collapsed
                h2.classList.add('collapsed');
                content.classList.add('collapsed');
                content.style.maxHeight = '0';
            } else {
                // Start expanded
                content.style.maxHeight = content.scrollHeight + 'px';
            }

            // Add click handler
            h2.addEventListener('click', function() {
                this.classList.toggle('collapsed');
                content.classList.toggle('collapsed');

                if (content.classList.contains('collapsed')) {
                    content.style.maxHeight = '0';
                } else {
                    content.style.maxHeight = content.scrollHeight + 'px';
                }
            });
        }
    });
});
"""


def get_report_css(theme: str = "slate_blue") -> str:
    """
    Get the complete CSS for a report with the specified theme.

    Parameters
    ----------
    theme : str, default="slate_blue"
        Color theme to use. Options:
        - "slate_blue": Muted blue-gray for compare-models reports
        - "teal": Green-teal for train reports
        - "taupe": Warm brown-gray for evaluate reports
        - "mauve": Muted purple for cross-dataset reports

    Returns
    -------
    str
        Complete CSS string including base styles and theme colors.
    """
    themes = {
        "slate_blue": CSS_SLATE_BLUE_THEME,
        "teal": CSS_TEAL_THEME,
        "taupe": CSS_TAUPE_THEME,
        "mauve": CSS_MAUVE_THEME,
    }
    theme_css = themes.get(theme, CSS_SLATE_BLUE_THEME)
    return REPORT_CSS_BASE + theme_css
