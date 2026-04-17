const { createApp, ref, computed, watch, nextTick } = Vue

createApp({
  setup() {
    const tab          = ref('result')
    const file         = ref(null)
    const preview      = ref(null)
    const previewExt   = ref('jpeg')
    const userText     = ref('')
    const isDrag       = ref(false)
    const emotion      = ref('')
    const probs        = ref({})
    const gifB64       = ref(null)
    const history      = ref([])
    const textMode     = ref('pick')
    const selectedText = ref('')
    const customText   = ref('')
    const textPool     = ref([])
    const textOffset   = ref(0)
    const loading      = ref({ predict: false, gen: false })
    const genProgress  = ref(0)
    const gifKey       = ref(0) // 强制重新渲染gif img
    const isDark       = ref(false) // 深色模式状态
    const imgId        = ref('')   // 当前上传图片的服务端 ID

    const COLORS = {
      '愤怒':'#f87171','开心':'#fbbf24','悲伤':'#60a5fa',
      '惊讶':'#fb923c','尴尬':'#86efac','中性':'#94a3b8','未知':'#a78bfa'
    }
    const ICONS = {
      '愤怒':'🔥','开心':'😄','悲伤':'😢',
      '惊讶':'😲','尴尬':'😅','中性':'😐','未知':'❓'
    }
    const BG = {
      '愤怒':'rgba(248,113,113,.06)','开心':'rgba(251,191,36,.06)',
      '悲伤':'rgba(96,165,250,.06)','惊讶':'rgba(251,146,60,.06)',
      '尴尬':'rgba(134,239,172,.06)','中性':'rgba(148,163,184,.05)','未知':'rgba(167,139,250,.06)'
    }
    const allEmotions = ['愤怒','开心','悲伤','惊讶','尴尬','中性','未知']

    const emotionColor = computed(() => COLORS[emotion.value] || '#94a3b8')
    const emotionIcon  = computed(() => ICONS[emotion.value]  || '🎭')
    const emotionBg    = computed(() => BG[emotion.value]     || 'rgba(148,163,184,.05)')
    const topProb      = computed(() => {
      const vals = Object.values(probs.value)
      if (!vals.length) return '0.0'
      return (Math.max(...vals) * 100).toFixed(1)
    })
    const textBatch = computed(() =>
      textPool.value.slice(textOffset.value, textOffset.value+6)
    )
    const finalText = computed(() => {
      if (textMode.value === 'none')   return ''
      if (textMode.value === 'custom') return customText.value
      return selectedText.value
    })

    // 文案加载（仅由 predict 主动调用，手动修正情绪不触发）
    async function loadTexts(val) {
      if (!val) return
      try {
        const r = await fetch(`/api/texts/${encodeURIComponent(val)}`)
        const d = await r.json()
        textPool.value     = d.texts || []
        textOffset.value   = 0
        selectedText.value = textPool.value[0] || ''
      } catch(e) {}
    }

    // 情绪变化：只触发入场动效
    watch(emotion, async val => {
      if (!val) return
      await nextTick()
      setTimeout(() => initObserver(), 100)
    })

    // GIF更新时触发loaded动效
    watch(gifB64, async val => {
      if (!val) return
      gifKey.value++
      await nextTick()
      setTimeout(() => {
        const img = document.querySelector('.gif-wrap img')
        if (img) {
          if (img.complete) {
            img.classList.add('loaded')
          } else {
            img.onload = () => img.classList.add('loaded')
          }
        }
      }, 50)
      setTimeout(() => initObserver(), 100)
    })

    // Toast 通知
    function showToast(msg, type = 'error') {
      const container = document.getElementById('toast-container')
      const el = document.createElement('div')
      el.className = `toast toast-${type}`
      el.textContent = (type === 'error' ? '⚠ ' : '✓ ') + msg
      container.appendChild(el)
      setTimeout(() => {
        el.classList.add('out')
        setTimeout(() => el.remove(), 300)
      }, 3500)
    }

    // IntersectionObserver 单例 — 整个生命周期只创建一次，避免泄漏
    const _observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible')
          _observer.unobserve(entry.target)
        }
      })
    }, { threshold: 0.12 })

    function initObserver() {
      document.querySelectorAll('.animate:not(.visible)').forEach(el => _observer.observe(el))
    }

    // 文件读取（上传后自动识别）
    async function readFile(f) {
      file.value = f
      const ext = f.name.split('.').pop().toLowerCase()
      previewExt.value = ext === 'jpg' ? 'jpeg' : ext
      const reader = new FileReader()
      reader.onload = ev => { preview.value = ev.target.result.split(',')[1] }
      reader.readAsDataURL(f)
      emotion.value = ''; probs.value = {}; gifB64.value = null
      await predict()
    }
    function onFile(e)  { if (e.target.files[0]) readFile(e.target.files[0]) }
    function onDrop(e)  { isDrag.value = false; if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]) }

    // API
    async function predict() {
      if (!file.value) return
      loading.value.predict = true
      try {
        const fd = new FormData()
        fd.append('image', file.value)
        fd.append('text', userText.value)
        const r = await fetch('/api/predict', { method:'POST', body:fd })
        const d = await r.json()
        if (d.error) { showToast(d.error); return }
        emotion.value    = d.emotion
        probs.value      = d.probs
        preview.value    = d.preview
        previewExt.value = d.ext || 'jpeg'
        imgId.value      = d.img_id || ''
        tab.value        = 'result'
        await loadTexts(d.emotion)
      } catch(e) { showToast('识别失败：' + e.message) }
      finally { loading.value.predict = false }
    }

    async function generate() {
      loading.value.gen = true
      gifB64.value = null
      genProgress.value = 0

      // 假进度：快速到60%，然后缓慢爬到92%，等待真实响应
      const ticks = [
        { target: 30, delay: 100 }, { target: 55, delay: 300 },
        { target: 72, delay: 600 }, { target: 85, delay: 1200 },
        { target: 92, delay: 2000 },
      ]
      const timers = []
      function runTicks(idx) {
        if (idx >= ticks.length) return
        timers.push(setTimeout(() => {
          genProgress.value = ticks[idx].target
          runTicks(idx + 1)
        }, ticks[idx].delay))
      }
      function clearAllTimers() { timers.forEach(clearTimeout); timers.length = 0 }
      runTicks(0)

      try {
        const r = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ emotion: emotion.value, text: finalText.value, img_id: imgId.value })
        })
        const d = await r.json()
        clearAllTimers()
        genProgress.value = 100
        if (d.error) { showToast(d.error); return }
        gifB64.value = d.gif
        showToast('表情包生成成功！', 'success')
        history.value.unshift({
          emotion: emotion.value,
          icon:    ICONS[emotion.value]  || '🎭',
          color:   COLORS[emotion.value] || '#94a3b8',
          gif:     d.gif,
          text:    finalText.value,
          ts:      new Date().toLocaleTimeString('zh-CN', { hour:'2-digit', minute:'2-digit' })
        })
        if (history.value.length > 20) history.value = history.value.slice(0,20)
      } catch(e) { clearAllTimers(); genProgress.value = 0; showToast('生成失败：' + e.message) }
      finally { loading.value.gen = false; setTimeout(() => { genProgress.value = 0 }, 600) }
    }

    function nextBatch() {
      textOffset.value   = (textOffset.value+6) % (textPool.value.length || 1)
      selectedText.value = textBatch.value[0] || ''
    }

    function reset() {
      file.value = null; preview.value = null
      emotion.value = ''; probs.value = {}
      gifB64.value = null; userText.value = ''; imgId.value = ''
    }

    // 深色模式切换
    function applyTheme(dark) {
      document.documentElement.classList.toggle('dark', dark)
    }
    function toggleTheme() {
      isDark.value = !isDark.value
      applyTheme(isDark.value)
      localStorage.setItem('darkMode', isDark.value)
    }

    // 挂载时恢复深色模式
    if (localStorage.getItem('darkMode') === 'true') {
      isDark.value = true
      applyTheme(true)
    }

    // 静态数据
    const stats = [
      { val:'80.12%', label:'独立测试集准确率' },
      { val:'20,000', label:'训练图片数量' },
      { val:'5 类',   label:'情感类别' },
      { val:'30 帧',  label:'GIF动效帧数' },
    ]
    const f1data = [
      { emo:'尴尬', ico:'😅', f1:.89, c:'#86efac' },
      { emo:'惊讶', ico:'😲', f1:.87, c:'#fb923c' },
      { emo:'开心', ico:'😄', f1:.85, c:'#fbbf24' },
      { emo:'愤怒', ico:'🔥', f1:.81, c:'#f87171' },
      { emo:'悲伤', ico:'😢', f1:.80, c:'#60a5fa' },
    ]
    const techStack = [
      { l:'数据层', i:'Bing爬虫 · FER2013 · EasyOCR · 数据增强' },
      { l:'模型层', i:'CLIP ViT-B/32 · CrossModalMLP · PyTorch' },
      { l:'生成层', i:'PIL · imageio · 5类差异化动效' },
      { l:'应用层', i:'Flask · Vue3 · 前后端分离架构' },
    ]

    return {
      tab, file, preview, previewExt, userText, isDrag,
      emotion, probs, gifB64, gifKey, history, textMode,
      selectedText, customText, textBatch, textOffset,
      loading, genProgress, COLORS, ICONS, allEmotions, isDark, imgId,
      emotionColor, emotionIcon, emotionBg, topProb, finalText,
      stats, f1data, techStack,
      onFile, onDrop, predict, generate, nextBatch, reset, toggleTheme
    }
  }
}).mount('#app')