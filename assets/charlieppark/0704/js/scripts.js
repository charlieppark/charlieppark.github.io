(function(){
  const TOTAL = 7;
  function getCompleted() {
    return JSON.parse(localStorage.getItem('completed') || '[]');
  }
  function markCompleted(id) {
    const arr = getCompleted();
    if (!arr.includes(id)) {
      arr.push(id);
      localStorage.setItem('completed', JSON.stringify(arr));
      renderShape();
    }
  }
  window.renderShape = function() {
    const done = getCompleted();
    document.querySelectorAll('[data-id]').forEach(el => {
      const id = parseInt(el.dataset.id,10);
      if (done.includes(id)) el.style.background='#000';
    });
  };
  window.checkAnswer = function(quizId, answer, correct, nextPage) {
    if (answer.trim() === correct) {
      markCompleted(quizId);
      window.location.href = nextPage;
    } else {
      alert('Incorrect, please try again.');
    }
  };
})();