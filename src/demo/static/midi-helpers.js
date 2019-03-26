let loadFile = () => {
};

document.addEventListener('DOMContentLoaded', function () {
    let tempoInput = document.getElementById('tempoInput');
    let playBtn = document.getElementById('playBtn');
    let visualizer = document.getElementById('visualizer');
    let file = document.getElementById('file');

    tempoInput.addEventListener('change', () => visualizer.tempo = tempoInput.value);
    playBtn.addEventListener('click', () => startOrStop());
    visualizer.addEventListener('visualizer-ready', () => {
        tempoInput.value = visualizer.tempo;
        playBtn.disabled = false;
        playBtn.textContent = 'play';
    });

    loadFile = function (e) {
        const file = e.target.files[0];
        visualizer.loadFile(file);
        return false;
    };

    function startOrStop() {
        if (visualizer.isPlaying()) {
            visualizer.stop();
            playBtn.textContent = 'play';
        } else {
            visualizer.tempo = tempoInput.value;
            visualizer.start();
            playBtn.textContent = 'stop';
        }
    }
});