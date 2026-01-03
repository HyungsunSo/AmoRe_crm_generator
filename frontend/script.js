/**
 * CRM Studio - Toss Style V2
 * Loads brand images from brand_images.json
 */

// Brand data - loaded from JSON
let BRAND_IMAGES = {};

const PERSONA_INFO = {
    'Luxury_Lover': { label: '프리미엄', color: '#9065B0', bg: '#F4F0F7' },
    'Sensitive_Skin': { label: '민감성', color: '#0F7B6C', bg: '#E6FCF5' },
    'Budget_Seeker': { label: '가성비', color: '#D9730D', bg: '#FFF4E6' },
    'Trend_Follower': { label: '트렌드', color: '#E03E8E', bg: '#FFF0F6' },
    'Natural_Beauty': { label: '자연주의', color: '#2383E2', bg: '#E7F5FF' }
};

let BRANDS_DATA = {}, PRODUCTS = [], PERSONAS = [], CAMPAIGN_EVENTS = {};

state = {
    currentStep: 1, selectedBrand: null, selectedProduct: null,
    stageIndex: null, styleIndex: null, selectedEvent: null, mode: 'simple',
    customData: { brandName: '', brandStory: '', productName: '', productPrice: 0 },
    customPersonas: [], customEvents: [], customStages: [], customStyles: []
};

let AARRR = ['Acquisition', 'Activation', 'Retention', 'Revenue', 'Referral'];
let AARRR_KR = ['유입', '구매', '재구매', '매출', '추천'];
let STYLES_KR = ['긴박', '정보', 'FOMO', '감성', '시즌'];

const $ = id => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
    loadData();
    setupEvents();
});

async function loadData() {
    try {
        const [prod, brand, persona, camp, brandImg] = await Promise.all([
            fetch('../data/products.json').then(r => r.json()),
            fetch('../data/brand_stories.json').then(r => r.json()),
            fetch('../data/personas.json').then(r => r.json()),
            fetch('../data/campaign_events.json').then(r => r.json()),
            fetch('./brand_images.json').then(r => r.json())
        ]);
        PRODUCTS = prod; BRANDS_DATA = brand; PERSONAS = persona; CAMPAIGN_EVENTS = camp; BRAND_IMAGES = brandImg;
        renderBrands();
    } catch (e) { console.error('Data load error:', e); renderBrands(); }
}

function setupEvents() {
    $('back-btn').onclick = prevStep;
    $('next-btn').onclick = nextStep;

    $('stage-chips').onclick = e => {
        if (!e.target.classList.contains('chip') || e.target.classList.contains('add-custom-btn')) return;
        document.querySelectorAll('#stage-chips .chip').forEach(c => c.classList.remove('active'));
        e.target.classList.add('active');
        state.stageIndex = +e.target.dataset.value;
        updateCampaignInfo(); updateSidebar();
    };

    $('style-chips').onclick = e => {
        if (!e.target.classList.contains('chip') || e.target.classList.contains('add-custom-btn')) return;
        document.querySelectorAll('#style-chips .chip').forEach(c => c.classList.remove('active'));
        e.target.classList.add('active');
        state.styleIndex = +e.target.dataset.value;
        updateSidebar();
    };

    document.addEventListener('click', e => {
        if (e.target.classList.contains('category-tab')) {
            document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            renderProducts(e.target.dataset.category);
        }
    });
}

function selectMode(mode) {
    state.mode = mode;
    $('mode-selection-screen').style.opacity = '0';
    setTimeout(() => {
        $('mode-selection-screen').style.display = 'none';
        $('main-app').style.display = 'grid';
        $('current-mode-badge').textContent = mode === 'simple' ? '간편 모드' : '전문가 모드';

        if (mode === 'expert') {
            $('simple-brand-view').style.display = 'none';
            $('expert-brand-view').style.display = 'block';
            $('expert-info-panel').style.display = 'block';
            $('expert-persona-creator').style.display = 'block';
            renderBrands(true);
        } else {
            $('simple-brand-view').style.display = 'block';
            $('expert-brand-view').style.display = 'none';
            renderBrands(false);
        }
    }, 200);
}

function goHome() { location.reload(); }

function renderBrands(isExpert = false) {
    const brands = Object.keys(BRANDS_DATA);
    const container = isExpert ? $('expert-brand-grid-list') : $('brand-grid');

    container.innerHTML = brands.map(b => {
        const info = BRAND_IMAGES[b] || {};
        const color = info.color || '#3182F6';
        const logo = info.logo_url || '';
        const eng = info.name_en || '';
        return `
            <div class="brand-card" data-brand="${b}" onclick="selectBrand('${b}')">
                <div class="brand-logo-circle" style="background: ${logo ? 'white' : color}; border: 1px solid #eee;">
                    ${logo ? `<img src="${logo}" alt="${b}" onerror="this.parentElement.innerHTML='${b[0]}'; this.parentElement.style.background='${color}'; this.parentElement.style.color='white';">` : `<span style="color:white;font-weight:700;font-size:18px;">${b[0]}</span>`}
                </div>
                <div class="brand-name">${b}</div>
                <div class="brand-name-en">${eng}</div>
            </div>
        `;
    }).join('');
}

function renderProducts(cat = 'all') {
    if (!state.selectedBrand) return;

    const brandProds = PRODUCTS.filter(p => p.brand_name === state.selectedBrand);
    const cats = [...new Set(brandProds.map(p => p.sub_category || p.category).filter(Boolean))];

    const tabsEl = state.mode === 'expert' ? $('expert-category-tabs') : $('category-tabs');
    tabsEl.innerHTML = `<button class="category-tab ${cat === 'all' ? 'active' : ''}" data-category="all">전체</button>` +
        cats.slice(0, 4).map(c => `<button class="category-tab ${cat === c ? 'active' : ''}" data-category="${c}">${c}</button>`).join('');

    const filtered = cat === 'all' ? brandProds : brandProds.filter(p => (p.sub_category || p.category) === cat);
    const listEl = state.mode === 'expert' ? $('expert-product-list') : $('product-list');

    if (!filtered.length) { listEl.innerHTML = '<p style="padding:20px;color:#888;">제품 없음</p>'; return; }

    listEl.innerHTML = filtered.slice(0, 12).map(p => {
        const img = p.image_urls?.[0] || '';
        const price = parseInt(p.price) || 0;
        return `
            <div class="product-item ${state.selectedProduct?.product_id === p.product_id ? 'selected' : ''}" onclick="selectProduct('${p.product_id}')">
                <div class="product-thumb">${img ? `<img src="${img}" onerror="this.style.display='none'">` : ''}</div>
                <div class="product-info">
                    <div class="product-name">${p.name}</div>
                    <div class="product-price">₩${price.toLocaleString()}</div>
                </div>
            </div>
        `;
    }).join('');
}

function updateCampaignInfo() {
    if (state.stageIndex === null) return;

    const stage = AARRR[state.stageIndex];
    const stageData = CAMPAIGN_EVENTS[stage];
    if (!stageData) return;

    // Get promotion events (promotion_y has promotions)
    const promoEvents = stageData.promotion_y || [];
    const nonPromoEvents = stageData.promotion_n || [];
    const allEvents = [...promoEvents, ...nonPromoEvents, ...state.customEvents];

    // Render event chips
    const eventChipsEl = $('event-chips');
    if (eventChipsEl) {
        eventChipsEl.innerHTML = allEvents.map((ev, i) => `
            <span class="chip ${state.selectedEvent?.id === ev.id ? 'active' : ''}" 
                  data-event-id="${ev.id}" onclick="selectEvent('${ev.id}')">${ev.name}</span>
        `).join('');
    }

    // Show event detail if selected
    const detailBox = $('event-detail-box');
    if (detailBox && state.selectedEvent) {
        detailBox.innerHTML = `
            <div class="info-row"><span class="label">이벤트</span><span class="val">${state.selectedEvent.name}</span></div>
            <div class="info-row"><span class="label">상세</span><span class="val desc">${state.selectedEvent.detail}</span></div>
        `;
        detailBox.style.display = 'block';
    } else if (detailBox) {
        detailBox.innerHTML = '<p style="color:#8B95A1;font-size:13px;">목적 선택 시 표시됩니다</p>';
    }
}

function selectEvent(eventId) {
    const stage = AARRR[state.stageIndex];
    const stageData = CAMPAIGN_EVENTS[stage];
    const allEvents = [...(stageData?.promotion_y || []), ...(stageData?.promotion_n || []), ...state.customEvents];
    state.selectedEvent = allEvents.find(e => e.id === eventId);
    updateCampaignInfo();
    updateSidebar();
}

function updateSidebar() {
    $('sidebar-brand').textContent = state.selectedBrand || '선택 필요';
    $('sidebar-product').textContent = state.selectedProduct?.name?.substring(0, 15) || '-';

    // Only show settings if they've been selected
    let settingsText = '-';
    if (state.stageIndex !== null && state.styleIndex !== null) {
        settingsText = `${AARRR_KR[state.stageIndex]} · ${STYLES_KR[state.styleIndex]}`;
    } else if (state.stageIndex !== null) {
        settingsText = AARRR_KR[state.stageIndex];
    }
    $('sidebar-settings').textContent = settingsText;

    document.querySelectorAll('.progress-step').forEach(s => {
        const n = +s.dataset.step;
        s.classList.toggle('active', n === state.currentStep);
        s.classList.toggle('completed', n < state.currentStep);
    });

    // Step 3 needs stage and style selected to proceed
    let canNext = false;
    if (state.currentStep === 1 && state.selectedBrand) canNext = true;
    if (state.currentStep === 2 && state.selectedProduct) canNext = true;
    if (state.currentStep === 3 && state.stageIndex !== null && state.styleIndex !== null) canNext = true;
    $('next-btn').disabled = !canNext;
    $('back-btn').disabled = state.currentStep === 1;

    // Update next button text
    $('next-btn').textContent = state.currentStep === 3 ? '메시지 생성' : '다음';
}

function selectBrand(b) {
    state.selectedBrand = b;
    document.querySelectorAll('.brand-card').forEach(c => c.classList.toggle('selected', c.dataset.brand === b));
    $('selected-brand-name').textContent = b;

    if (state.mode === 'expert') {
        $('custom-brand-name').value = b;
        $('custom-brand-story').value = BRANDS_DATA[b]?.story || '';
    }
    updateSidebar();
}

function selectProduct(id) {
    state.selectedProduct = PRODUCTS.find(p => p.product_id === id);

    if (state.mode === 'expert' && state.selectedProduct) {
        $('custom-product-name').value = state.selectedProduct.name;
        $('custom-product-price').value = state.selectedProduct.price;
    }

    renderProducts(document.querySelector('.category-tab.active')?.dataset.category || 'all');
    updateSidebar();
}

function nextStep() {
    if (state.currentStep === 3) { generateMessages(); return; }
    goToStep(state.currentStep + 1);
}

function prevStep() { if (state.currentStep > 1) goToStep(state.currentStep - 1); }

function goToStep(step) {
    document.querySelectorAll('.step-section').forEach((el, i) => el.classList.toggle('active', i === step - 1));
    state.currentStep = step;

    if (step === 2 && state.selectedBrand) {
        if (state.mode === 'expert') {
            $('expert-product-view').style.display = 'block';
            $('simple-product-view').style.display = 'none';
        } else {
            $('expert-product-view').style.display = 'none';
            $('simple-product-view').style.display = 'block';
        }
        renderProducts('all');
    }
    if (step === 3) {
        // Show expert creator panel in expert mode
        if (state.mode === 'expert') {
            $('expert-persona-creator').style.display = 'block';
        } else {
            $('expert-persona-creator').style.display = 'none';
        }
        updateCampaignInfo();
    }
    updateSidebar();
}

async function generateMessages() {
    const overlay = $('loading-overlay');
    overlay.style.display = 'flex';

    const steps = ['load-step-1', 'load-step-2', 'load-step-3'];
    for (let i = 0; i < steps.length; i++) {
        steps.forEach((s, j) => {
            $(s).classList.toggle('active', j === i);
            $(s).classList.toggle('done', j < i);
        });
        await new Promise(r => setTimeout(r, 800));
    }
    steps.forEach(s => { $(s).classList.remove('active'); $(s).classList.add('done'); });

    overlay.style.display = 'none';
    renderPhoneMockups();
    goToStep(4);
}

function renderPhoneMockups() {
    const brand = state.selectedBrand;
    const info = BRAND_IMAGES[brand] || {};
    const color = info.color || '#3182F6';
    const logo = info.logo_url || '';
    const product = state.selectedProduct?.name || '제품';
    const price = state.selectedProduct?.price ? parseInt(state.selectedProduct.price).toLocaleString() : '';
    const eventName = state.selectedEvent?.name || '';

    // Realistic persona-specific messages
    const personaMessages = {
        'Luxury_Lover': {
            title: `[${brand}] VIP 고객님을 위한 특별 제안`,
            body: `프리미엄 ${product}을 먼저 만나보세요.\n지금 구매 시 럭셔리 샘플 3종 증정 💎`
        },
        'Sensitive_Skin': {
            title: `[${brand}] 민감 피부를 위한 맞춤 케어`,
            body: `${product}은 저자극 테스트 완료 제품입니다.\n피부과 전문의 추천 포뮬러로 순하게 케어하세요 🌿`
        },
        'Budget_Seeker': {
            title: `[${brand}] ${eventName || '오늘만'} 특가!`,
            body: `${product} ${price ? `정가 ₩${price}` : ''}\n지금 20% 할인 + 무료배송 ✨`
        },
        'Trend_Follower': {
            title: `[${brand}] 인플루언서 PICK 🔥`,
            body: `SNS에서 핫한 ${product}!\n#틱톡바이럴 #올영픽 지금 품절 전 득템하세요`
        },
        'Natural_Beauty': {
            title: `[${brand}] 자연이 선사하는 아름다움`,
            body: `${product}에 담긴 청정 자연 성분.\n비건 인증 제품으로 건강한 뷰티 루틴을 완성하세요 🌱`
        }
    };

    const msgs = PERSONAS.map(p => ({
        name: p.name,
        ...PERSONA_INFO[p.name],
        ...(personaMessages[p.name] || { title: `[${brand}] 특별 혜택`, body: `${product} 지금 확인하세요` })
    }));

    // Add custom personas
    state.customPersonas.forEach(p => {
        msgs.push({
            name: p.name,
            label: '커스텀',
            bg: '#F4F0F7',
            color: '#9065B0',
            title: `[${brand}] ${p.name}님께 드리는 제안`,
            body: `${product} ${p.keywords ? `(${p.keywords})` : ''}\n특별한 혜택을 확인해보세요 ✨`
        });
    });

    $('phone-carousel').innerHTML = msgs.map(m => `
        <div class="phone-mockup">
            <div class="iphone-frame">
                <div class="iphone-screen">
                    <div style="text-align:right;color:#fff;font-size:13px;margin-bottom:60px;padding-right:8px;">9:41</div>
                    <div class="notif-card">
                        <div class="notif-header">
                            <div class="notif-icon" style="background:${color};">${logo ? `<img src="${logo}" style="width:14px;height:14px;">` : brand[0]}</div>
                            <span class="notif-app">${brand}</span>
                            <span class="notif-time">지금</span>
                        </div>
                        <div class="notif-title">${m.title}</div>
                        <div class="notif-body">${m.body.replace(/\n/g, '<br>')}</div>
                        <div class="persona-badge" style="background:${m.bg};color:${m.color};">${m.label}</div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

function resetWizard() {
    state = { currentStep: 1, selectedBrand: null, selectedProduct: null, stageIndex: null, styleIndex: null, selectedEvent: null, mode: state.mode, customData: {} };
    document.querySelectorAll('.brand-card').forEach(c => c.classList.remove('selected'));
    goToStep(1);
}

function exportAll() {
    const data = {
        brand: state.selectedBrand, product: state.selectedProduct,
        stage: AARRR[state.stageIndex], personas: PERSONAS.map(p => p.name),
        generated: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `crm_${state.selectedBrand}_${Date.now()}.json`;
    a.click();
}

function useCustomBrand() { state.selectedBrand = $('custom-brand-name').value || '커스텀'; updateSidebar(); }
function useCustomProduct() { state.selectedProduct = { name: $('custom-product-name').value, price: $('custom-product-price').value }; updateSidebar(); }
function addCustomPersona() {
    const name = $('new-persona-name').value.trim();
    const keywords = $('new-persona-keywords').value.trim();
    if (!name) { alert('페르소나 이름을 입력해주세요'); return; }

    const id = 'custom_p_' + Date.now();
    const newPersona = { id, name, keywords };
    state.customPersonas.push(newPersona);

    // Clear inputs
    $('new-persona-name').value = '';
    $('new-persona-keywords').value = '';

    renderAddedPersonas();
}

function removeCustomPersona(id) {
    state.customPersonas = state.customPersonas.filter(p => p.id !== id);
    renderAddedPersonas();
}

function renderAddedPersonas() {
    const listEl = $('added-personas-list');
    listEl.innerHTML = state.customPersonas.map(p => `
        <div class="added-item-tag">
            <span>${p.name}</span>
            <span class="remove-btn" onclick="removeCustomPersona('${p.id}')">×</span>
        </div>
    `).join('');
}

function addCustomEvent() {
    const name = $('new-event-name').value.trim();
    const detail = $('new-event-detail').value.trim();
    if (!name) { alert('이벤트 이름을 입력해주세요'); return; }

    const id = 'custom_ev_' + Date.now();
    const newEvent = { id, name, detail };
    state.customEvents.push(newEvent);

    // Clear inputs
    $('new-event-name').value = '';
    $('new-event-detail').value = '';

    renderAddedEvents();
}

function removeCustomEvent(id) {
    state.customEvents = state.customEvents.filter(e => e.id !== id);
    renderAddedEvents();
}
function addCustomStage() {
    const name = $('new-stage-name').value.trim();
    const detail = $('new-stage-detail').value.trim();
    if (!name) { alert('발송 목적 이름을 입력해주세요'); return; }

    const id = 'custom_st_' + Date.now();
    const newStage = { id, name, detail };
    state.customStages.push(newStage);

    // Add to selection arrays
    const newIndex = AARRR_KR.length;
    AARRR_KR.push(name);
    AARRR.push('Custom');

    // Add UI Chip
    const chipsContainer = $('stage-chips');
    // Insert before the "+ 직접 입력" button
    const addButton = chipsContainer.querySelector('.add-custom-btn');
    const newBtn = document.createElement('button');
    newBtn.className = 'chip';
    newBtn.dataset.value = newIndex;
    newBtn.dataset.customId = id; // Track ID
    newBtn.textContent = name;
    chipsContainer.insertBefore(newBtn, addButton);

    // Clear inputs
    $('new-stage-name').value = '';
    $('new-stage-detail').value = '';

    renderAddedStages();
    // Auto select
    newBtn.click();
}

function removeCustomStage(id) {
    state.customStages = state.customStages.filter(s => s.id !== id);
    // Note: Removing from AARRR/AARRR_KR and chips is complex due to index shift.
    // For simplicity in this demo, we allow adding but removal only removes from the "Added List" display and state.
    // Ideally, we'd rebuild the chips, but indices are hardcoded in data-value.
    renderAddedStages();
}

function renderAddedStages() {
    const listEl = $('added-stages-list');
    listEl.innerHTML = state.customStages.map(s => `
        <div class="added-item-tag">
            <span>${s.name}</span>
            <span class="remove-btn" onclick="removeCustomStage('${s.id}')">×</span>
        </div>
    `).join('');
}

function addCustomStyle() {
    const name = $('new-style-name').value.trim();
    const detail = $('new-style-detail').value.trim();
    if (!name) { alert('스타일 이름을 입력해주세요'); return; }

    const id = 'custom_sy_' + Date.now();
    const newStyle = { id, name, detail };
    state.customStyles.push(newStyle);

    // Add to selection arrays
    const newIndex = STYLES_KR.length;
    STYLES_KR.push(name);

    // Add UI Chip
    const chipsContainer = $('style-chips');
    const addButton = chipsContainer.querySelector('.add-custom-btn');
    const newBtn = document.createElement('button');
    newBtn.className = 'chip';
    newBtn.dataset.value = newIndex;
    newBtn.dataset.customId = id;
    newBtn.textContent = name;
    chipsContainer.insertBefore(newBtn, addButton);

    // Clear inputs
    $('new-style-name').value = '';
    $('new-style-detail').value = '';

    renderAddedStyles();
    // Auto select
    newBtn.click();
}

function removeCustomStyle(id) {
    state.customStyles = state.customStyles.filter(s => s.id !== id);
    renderAddedStyles();
}

function renderAddedStyles() {
    const listEl = $('added-styles-list');
    listEl.innerHTML = state.customStyles.map(s => `
        <div class="added-item-tag">
            <span>${s.name}</span>
            <span class="remove-btn" onclick="removeCustomStyle('${s.id}')">×</span>
        </div>
    `).join('');
}
function renderAddedEvents() {
    const listEl = $('added-events-list');
    const chipsEl = $('event-chips');

    // Render in the creator list
    listEl.innerHTML = state.customEvents.map(e => `
        <div class="added-item-tag">
            <span>${e.name}</span>
            <span class="remove-btn" onclick="removeCustomEvent('${e.id}')">×</span>
        </div>
    `).join('');

    // Also update the selection chips if in step 3
    if (state.currentStep === 3) {
        updateCampaignInfo();
    }
}
