// Main Application Logic for Big Five Personality Assessment

class PersonalityAssessment {
    constructor() {
        this.currentQuestion = 0;
        this.answers = {};
        this.studentInfo = {};
        this.questions = shuffleArray(personalityQuestions);
        this.chart = null;

        this.initializeApp();
    }

    initializeApp() {
        // Screen elements
        this.welcomeScreen = document.getElementById('welcome-screen');
        this.infoScreen = document.getElementById('info-screen');
        this.quizScreen = document.getElementById('quiz-screen');
        this.resultsScreen = document.getElementById('results-screen');

        // Buttons
        this.startBtn = document.getElementById('start-btn');
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.retakeBtn = document.getElementById('retake-btn');
        this.downloadBtn = document.getElementById('download-btn');

        // Quiz elements
        this.questionText = document.getElementById('question-text');
        this.traitIndicator = document.getElementById('trait-indicator');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.scaleOptions = document.getElementById('scale-options');

        // Bind events
        this.bindEvents();
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.showInfoScreen());

        document.getElementById('student-info-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.collectStudentInfo();
            this.showQuizScreen();
        });

        this.prevBtn.addEventListener('click', () => this.previousQuestion());
        this.nextBtn.addEventListener('click', () => this.nextQuestion());
        this.retakeBtn.addEventListener('click', () => this.retakeAssessment());
        this.downloadBtn.addEventListener('click', () => this.downloadResults());

        // Scale button events
        const scaleBtns = this.scaleOptions.querySelectorAll('.scale-btn');
        scaleBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.selectAnswer(e.target.dataset.value));
        });
    }

    showScreen(screen) {
        [this.welcomeScreen, this.infoScreen, this.quizScreen, this.resultsScreen].forEach(s => {
            s.classList.remove('active');
        });
        screen.classList.add('active');
    }

    showInfoScreen() {
        this.showScreen(this.infoScreen);
    }

    collectStudentInfo() {
        this.studentInfo = {
            name: document.getElementById('student-name').value || 'Student',
            age: document.getElementById('student-age').value || '',
            grade: document.getElementById('student-grade').value || ''
        };
    }

    showQuizScreen() {
        this.showScreen(this.quizScreen);
        this.displayQuestion();
    }

    displayQuestion() {
        const question = this.questions[this.currentQuestion];

        // Update question text
        this.questionText.textContent = question.text;

        // Update trait indicator
        const traitNames = {
            'O': 'Openness',
            'C': 'Conscientiousness',
            'E': 'Extraversion',
            'A': 'Agreeableness',
            'N': 'Neuroticism'
        };
        this.traitIndicator.textContent = traitNames[question.trait];
        this.traitIndicator.className = `trait-indicator trait-${question.trait}`;

        // Update progress
        const progress = ((this.currentQuestion + 1) / this.questions.length) * 100;
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `Question ${this.currentQuestion + 1} of ${this.questions.length}`;

        // Update scale buttons
        const scaleBtns = this.scaleOptions.querySelectorAll('.scale-btn');
        scaleBtns.forEach(btn => {
            btn.classList.remove('selected');
            if (this.answers[question.id] && btn.dataset.value == this.answers[question.id]) {
                btn.classList.add('selected');
            }
        });

        // Update navigation buttons
        this.prevBtn.disabled = this.currentQuestion === 0;
        this.nextBtn.disabled = !this.answers[question.id];

        // Update next button text
        if (this.currentQuestion === this.questions.length - 1) {
            this.nextBtn.textContent = 'See Results';
        } else {
            this.nextBtn.textContent = 'Next';
        }
    }

    selectAnswer(value) {
        const question = this.questions[this.currentQuestion];
        this.answers[question.id] = parseInt(value);

        // Update button styles
        const scaleBtns = this.scaleOptions.querySelectorAll('.scale-btn');
        scaleBtns.forEach(btn => {
            btn.classList.remove('selected');
            if (btn.dataset.value == value) {
                btn.classList.add('selected');
            }
        });

        // Enable next button
        this.nextBtn.disabled = false;
    }

    previousQuestion() {
        if (this.currentQuestion > 0) {
            this.currentQuestion--;
            this.displayQuestion();
        }
    }

    nextQuestion() {
        if (this.currentQuestion < this.questions.length - 1) {
            this.currentQuestion++;
            this.displayQuestion();
        } else {
            this.calculateResults();
        }
    }

    calculateResults() {
        const scores = {
            O: { total: 0, count: 0 },
            C: { total: 0, count: 0 },
            E: { total: 0, count: 0 },
            A: { total: 0, count: 0 },
            N: { total: 0, count: 0 }
        };

        // Calculate scores for each trait
        this.questions.forEach(question => {
            let score = this.answers[question.id];

            // Reverse score if needed (1->5, 2->4, 3->3, 4->2, 5->1)
            if (question.reversed) {
                score = 6 - score;
            }

            scores[question.trait].total += score;
            scores[question.trait].count++;
        });

        // Convert to percentages (0-100)
        this.results = {};
        for (let trait in scores) {
            // Max possible score is 50 (10 questions * 5 points each)
            const maxScore = scores[trait].count * 5;
            const percentage = (scores[trait].total / maxScore) * 100;
            this.results[trait] = Math.round(percentage);
        }

        this.showResults();
    }

    showResults() {
        this.showScreen(this.resultsScreen);

        // Update greeting
        const greeting = document.getElementById('student-greeting');
        greeting.textContent = `Here are your results, ${this.studentInfo.name}! Your personality profile based on the Big Five model:`;

        // Create radar chart
        this.createChart();

        // Display detailed results
        this.displayDetailedResults();
    }

    createChart() {
        const ctx = document.getElementById('personality-chart').getContext('2d');

        if (this.chart) {
            this.chart.destroy();
        }

        this.chart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: [
                    'Openness',
                    'Conscientiousness',
                    'Extraversion',
                    'Agreeableness',
                    'Neuroticism'
                ],
                datasets: [{
                    label: 'Your Personality Profile',
                    data: [
                        this.results.O,
                        this.results.C,
                        this.results.E,
                        this.results.A,
                        this.results.N
                    ],
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 3,
                    pointBackgroundColor: [
                        '#f56565',
                        '#48bb78',
                        '#ed8936',
                        '#4299e1',
                        '#9f7aea'
                    ],
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(102, 126, 234, 1)',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        min: 0,
                        ticks: {
                            stepSize: 20,
                            font: {
                                size: 12
                            }
                        },
                        pointLabels: {
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    displayDetailedResults() {
        const container = document.getElementById('trait-results');
        container.innerHTML = '';

        const traitOrder = ['O', 'C', 'E', 'A', 'N'];

        traitOrder.forEach(trait => {
            const score = this.results[trait];
            const info = traitInfo[trait];
            const level = this.getLevel(score);
            const levelText = level.charAt(0).toUpperCase() + level.slice(1);

            let description;
            if (score >= 70) {
                description = info.high;
            } else if (score >= 40) {
                description = info.medium;
            } else {
                description = info.low;
            }

            const html = `
                <div class="trait-result-item trait-result-${trait}">
                    <h4>
                        ${info.name}
                        <span class="level level-${level}">${levelText}</span>
                    </h4>
                    <div class="score">Score: ${score}%</div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${score}%"></div>
                    </div>
                    <div class="description">${description}</div>
                </div>
            `;

            container.innerHTML += html;
        });
    }

    getLevel(score) {
        if (score >= 70) return 'high';
        if (score >= 40) return 'medium';
        return 'low';
    }

    retakeAssessment() {
        // Reset everything
        this.currentQuestion = 0;
        this.answers = {};
        this.questions = shuffleArray(personalityQuestions);

        if (this.chart) {
            this.chart.destroy();
        }

        this.showScreen(this.welcomeScreen);
    }

    downloadResults() {
        // Create a text summary of results
        let summary = `Big Five Personality Assessment Results\n`;
        summary += `=====================================\n\n`;
        summary += `Student: ${this.studentInfo.name}\n`;
        if (this.studentInfo.age) summary += `Age: ${this.studentInfo.age}\n`;
        if (this.studentInfo.grade) summary += `Grade: ${this.studentInfo.grade}\n`;
        summary += `Date: ${new Date().toLocaleDateString()}\n\n`;
        summary += `PERSONALITY SCORES:\n`;
        summary += `-------------------\n`;

        const traitOrder = ['O', 'C', 'E', 'A', 'N'];
        traitOrder.forEach(trait => {
            const score = this.results[trait];
            const info = traitInfo[trait];
            const level = this.getLevel(score);
            summary += `\n${info.name}: ${score}% (${level.toUpperCase()})\n`;

            let description;
            if (score >= 70) {
                description = info.high;
            } else if (score >= 40) {
                description = info.medium;
            } else {
                description = info.low;
            }
            summary += `${description}\n`;
        });

        summary += `\n=====================================\n`;
        summary += `This assessment is for educational purposes only.\n`;
        summary += `Based on the Big Five/OCEAN model of personality.\n`;

        // Create and download file
        const blob = new Blob([summary], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `personality-results-${this.studentInfo.name.replace(/\s+/g, '-')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PersonalityAssessment();
});
