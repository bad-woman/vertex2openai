"""
无头浏览器管理模块

使用 Playwright 管理无头 Chromium 浏览器实例。
已登录账号模式 - 最小化反检测配置。
"""

import asyncio
from typing import Optional, Callable
from pathlib import Path

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class HeadlessBrowser:
    """无头浏览器管理器 - 已登录账号模式"""

    # Vertex AI Studio URL
    VERTEX_AI_URL = "https://console.cloud.google.com/vertex-ai/studio/multimodal?mode=prompt"

    # 用户数据目录 (保存登录态)
    USER_DATA_DIR = "config/browser_data"

    def __init__(self):
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._is_running = False
        self.needs_manual_login = False

    @staticmethod
    def check_availability() -> bool:
        """检查 Playwright 是否可用"""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright 未安装，请运行: pip install playwright && playwright install chromium")
            return False
        return True

    async def start(self, headless: bool = True) -> bool:
        """
        启动浏览器

        Args:
            headless: 是否无头模式 (首次登录时设为 False)
        """
        if not self.check_availability():
            return False

        try:
            print("🌐 正在启动浏览器...")

            # 确保用户数据目录存在
            user_data_path = Path(self.USER_DATA_DIR)
            user_data_path.mkdir(parents=True, exist_ok=True)

            self.playwright = await async_playwright().start()

            # 最小化启动参数 - 已登录账号无需复杂反检测
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-extensions",
            ]

            # 使用持久化上下文保留登录态
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(user_data_path),
                headless=headless,
                viewport={"width": 1920, "height": 1080},
                args=launch_args,
                ignore_default_args=["--enable-automation"],
                locale="en-US",
            )

            # 获取或创建页面
            if self.context.pages:
                self.page = self.context.pages[0]
            else:
                self.page = await self.context.new_page()

            # 注入最小化反检测脚本
            await self._inject_stealth_script()

            self._is_running = True
            print("✅ 浏览器已启动")
            return True

        except Exception as e:
            print(f"❌ 浏览器启动失败: {e}")
            return False

    async def _inject_stealth_script(self) -> None:
        """注入最小化反检测脚本 - 仅隐藏 webdriver 标志"""
        if not self.context:
            return

        stealth_js = '''
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
            configurable: true
        });
        window.chrome = {
            runtime: {
                onConnect: { addListener: () => {}, removeListener: () => {} },
                onMessage: { addListener: () => {}, removeListener: () => {} },
                sendMessage: () => {},
                connect: () => ({ onMessage: { addListener: () => {} }, postMessage: () => {}, disconnect: () => {} })
            }
        };
        '''
        await self.context.add_init_script(stealth_js)
        print("🛡️ 反检测脚本已注入")

    async def navigate_to_vertex(self) -> bool:
        """
        导航到 Vertex AI Studio
        """
        if not self.page:
            print("❌ 浏览器未启动")
            return False

        try:
            print("🔗 正在导航到 Vertex AI Studio...")

            # 注入云端环境提供的 Cookie (如果有)
            from config import GOOGLE_COOKIE
            from runtime_state import app_state
            
            cookie_str = GOOGLE_COOKIE or app_state.get_google_cookie()
            if cookie_str and self.context:
                print("🍪 检测到预设 Google Cookie，尝试免登录注入...")
                cookies = []
                for pair in cookie_str.split(';'):
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        if not k: continue
                        cookies.append({
                            "name": k,
                            "value": v,
                            "domain": ".google.com",
                            "path": "/"
                        })
                if cookies:
                    try:
                        await self.context.add_cookies(cookies)
                    except Exception as e:
                        print(f"❌ 注入 Cookie 失败: {e}")

            try:
                await self.page.goto(self.VERTEX_AI_URL, wait_until="domcontentloaded", timeout=30000)

                # 检查是否需要登录
                current_url = self.page.url
                if "accounts.google.com" in current_url or "signin" in current_url:
                    if cookie_str:
                        print("⚠️ 预设的 Cookie 可能已失效，被重定向到登录页")
                    print("⚠️ 需要登录 Google 账号")
                    print("   检测到需要登录，已进入云端交互登录模式。请打开大盘控制台进行远程可视化登录！")
                    self.needs_manual_login = True
                    try:
                        # 循环等待直到离开登录页
                        while "accounts.google.com" in self.page.url or "signin" in self.page.url:
                            await asyncio.sleep(1)
                        print("✅ 登录成功，已离开登录页")
                    except Exception as e:
                        print(f"❌ 等待登录遇到错误: {e}")
                    finally:
                        self.needs_manual_login = False

                # 等待页面进一步加载稳定
                await asyncio.sleep(3)

            except Exception as e:
                print(f"⚠️ 初始导航遇到问题: {e}")

            # 检测并处理条款弹窗
            await self._accept_terms_if_present()

            print("✅ 已到达 Vertex AI Studio")
            return True

        except Exception as e:
            print(f"❌ 导航失败: {e}")
            return False

    async def _accept_terms_if_present(self) -> bool:
        """自动检测并同意条款弹窗（简化版 - 单次 evaluate 调用）"""
        if not self.page:
            return False

        try:
            result = await self.page.evaluate('''() => {
                // 1. 检查是否存在条款对话框
                const dialogSelectors = ['[role="dialog"]', '.mdc-dialog', 'p.notranslate', '[aria-modal="true"]'];
                let dialogFound = false;

                for (const sel of dialogSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) {
                        const text = (el.textContent || '').toLowerCase();
                        if (text.includes('terms') || text.includes('agree') ||
                            text.includes('条款') || text.includes('同意') ||
                            text.includes('consent') || text.includes('accept')) {
                            dialogFound = true;
                            break;
                        }
                    }
                }

                if (!dialogFound) return 'no_terms';

                // 2. 滚动条款内容到底部
                const scrollContainers = document.querySelectorAll(
                    '.mdc-dialog__content, [role="dialog"] [style*="overflow"]'
                );
                scrollContainers.forEach(c => {
                    const style = window.getComputedStyle(c);
                    if (style.overflow === 'auto' || style.overflow === 'scroll' ||
                        style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        c.scrollTop = c.scrollHeight;
                    }
                });
                const termsText = document.querySelector('p.notranslate');
                if (termsText) termsText.scrollIntoView({ block: 'end' });

                // 3. 勾选复选框
                const checkboxSelectors = [
                    'input.mdc-checkbox__native-control[type="checkbox"]',
                    '[role="dialog"] input[type="checkbox"]',
                    '.mdc-checkbox input[type="checkbox"]'
                ];
                for (const sel of checkboxSelectors) {
                    const cb = document.querySelector(sel);
                    if (cb && !cb.checked) {
                        cb.click();
                        break;
                    }
                }

                // 4. 点击同意按钮
                const buttonSelectors = [
                    'button:has(span.mdc-button__label)',
                    'button[type="submit"]',
                    '.mdc-dialog__actions button:last-child'
                ];
                const agreeKeywords = ['同意', 'agree', 'accept'];

                for (const sel of buttonSelectors) {
                    const buttons = document.querySelectorAll(sel);
                    for (const btn of buttons) {
                        const btnText = (btn.textContent || '').toLowerCase().trim();
                        if (agreeKeywords.some(k => btnText.includes(k)) && !btn.disabled) {
                            btn.click();
                            return 'accepted';
                        }
                    }
                }

                return 'button_not_found';
            }''')

            if result == 'accepted':
                print("✅ 条款已自动同意")
                await asyncio.sleep(0.3)
                return True
            elif result == 'no_terms':
                print("ℹ️ 未检测到条款对话框")
                return True
            else:
                print("⚠️ 检测到条款但未找到同意按钮")
                return False

        except Exception as e:
            print(f"⚠️ 自动同意条款失败: {e}")
            return False

    async def setup_request_interception(self, on_request: Callable) -> None:
        """
        设置请求拦截

        Args:
            on_request: 请求回调函数 (async callable accepting Playwright request)
        """
        if not self.page:
            return

        async def handle_request(request):
            url = request.url
            # 只关注 Vertex AI 相关请求
            if "batchGraphql" in url or "StreamGenerateContent" in url:
                await on_request(request)

        self.page.on("request", handle_request)
        print("🔍 请求拦截已设置")

    async def send_test_message(self, max_retries: int = 3) -> bool:
        """
        发送测试消息触发 API 请求，用于获取/刷新凭证

        Args:
            max_retries: 最大重试次数

        Returns:
            是否成功发送
        """
        if not self.page:
            return False

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"💬 重试发送测试消息 ({attempt + 1}/{max_retries})...")
                else:
                    print("💬 正在发送测试消息...")

                # 1. 先关闭任何可能存在的 overlay 遮罩层
                await self._dismiss_overlays()

                # 2. 等待输入框出现
                input_selector = (
                    'textarea[aria-label*="message"], '
                    'div[contenteditable="true"], '
                    'textarea[placeholder*="message"], '
                    'textarea[placeholder*="消息"]'
                )
                try:
                    await self.page.wait_for_selector(input_selector, timeout=10000)
                except Exception:
                    if attempt < max_retries - 1:
                        print("   ⚠️ 输入框未出现，重试中...")
                        await asyncio.sleep(2)
                        continue
                    raise

                # 3. 使用 JavaScript 直接聚焦和输入（绕过 overlay 问题）
                success = await self.page.evaluate('''() => {
                    // 关闭所有 overlay
                    const overlays = document.querySelectorAll('.cdk-overlay-backdrop, .cdk-overlay-container > *');
                    overlays.forEach(el => {
                        if (el.classList.contains('cdk-overlay-backdrop')) {
                            el.click();
                        }
                    });

                    // 查找输入框
                    const selectors = [
                        'textarea[aria-label*="message"]',
                        'div[contenteditable="true"]',
                        'textarea[placeholder*="message"]',
                        'textarea[placeholder*="消息"]'
                    ];

                    let input = null;
                    for (const sel of selectors) {
                        input = document.querySelector(sel);
                        if (input && input.offsetParent !== null) break;
                        input = null;
                    }

                    if (!input) return false;

                    // 聚焦输入框
                    input.focus();

                    // 设置内容
                    if (input.tagName === 'TEXTAREA') {
                        input.value = 'hi';
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    } else {
                        // contenteditable
                        input.textContent = 'hi';
                        input.dispatchEvent(new InputEvent('input', { bubbles: true, data: 'hi' }));
                    }

                    return true;
                }''')

                if not success:
                    if attempt < max_retries - 1:
                        print("   ⚠️ 无法设置输入内容，重试中...")
                        await asyncio.sleep(1)
                        continue
                    print("❌ 未找到可用的输入框")
                    return False

                await asyncio.sleep(0.1)

                # 4. 按回车发送
                await self.page.keyboard.press("Enter")
                print("✅ 测试消息已发送")
                return True

            except Exception as e:
                error_msg = str(e)
                if "intercepts pointer events" in error_msg and attempt < max_retries - 1:
                    print("   ⚠️ 检测到 overlay 遮挡，尝试关闭...")
                    await self._dismiss_overlays()
                    await asyncio.sleep(0.5)
                    continue
                elif attempt < max_retries - 1:
                    print(f"   ⚠️ 发送失败: {error_msg[:50]}，重试中...")
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"❌ 发送消息失败: {e}")
                    return False

        return False

    async def _dismiss_overlays(self) -> None:
        """关闭页面上的 overlay 遮罩层"""
        if not self.page:
            return

        try:
            await self.page.evaluate('''() => {
                // 1. 点击所有 backdrop 关闭对话框
                const backdrops = document.querySelectorAll('.cdk-overlay-backdrop');
                backdrops.forEach(backdrop => {
                    if (backdrop.offsetParent !== null) {
                        backdrop.click();
                    }
                });

                // 2. 按 Escape 键关闭任何模态
                document.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Escape',
                    code: 'Escape',
                    keyCode: 27,
                    which: 27,
                    bubbles: true
                }));

                // 3. 尝试点击 overlay 容器中的关闭按钮
                const overlayContainer = document.querySelector('.cdk-overlay-container');
                if (overlayContainer) {
                    const activeBackdrop = overlayContainer.querySelector('.cdk-overlay-backdrop-showing');
                    if (activeBackdrop) {
                        const closeButtons = overlayContainer.querySelectorAll(
                            'button[aria-label*="close"], button[aria-label*="Close"], ' +
                            'button[aria-label*="关闭"], .mat-dialog-close, ' +
                            'button.close, [mat-dialog-close]'
                        );
                        closeButtons.forEach(btn => btn.click());
                    }
                }
            }''')

            # 等待 overlay 动画完成
            await asyncio.sleep(0.3)

        except Exception as e:
            print(f"   ⚠️ 关闭 overlay 时出错: {e}")

    async def close(self) -> None:
        """关闭浏览器"""
        self._is_running = False
        self.needs_manual_login = False
        if self.context:
            await self.context.close()
            self.context = None
            self.page = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        print("🔒 浏览器已关闭")

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ========== 远程可视化交互接口 ==========

    async def get_screenshot(self) -> Optional[bytes]:
        """获取当前页面截图 (JPEG格式)"""
        if not self.page or not self.is_running:
            return None
        try:
            return await self.page.screenshot(type="jpeg", quality=60)
        except Exception:
            return None

    async def send_click(self, x: float, y: float) -> bool:
        """发送鼠标点击事件"""
        if not self.page or not self.is_running:
            return False
        try:
            await self.page.mouse.click(x, y)
            return True
        except Exception:
            return False

    async def send_text(self, text: str) -> bool:
        """发送文本输入 (模拟键盘)"""
        if not self.page or not self.is_running:
            return False
        try:
            await self.page.keyboard.insert_text(text)
            return True
        except Exception:
            return False

    async def send_key(self, key: str) -> bool:
        """发送按键 (如 Enter)"""
        if not self.page or not self.is_running:
            return False
        try:
            await self.page.keyboard.press(key)
            return True
        except Exception:
            return False
