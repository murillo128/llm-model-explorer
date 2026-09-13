import js from '@eslint/js';
import globals from 'globals';
import hooks from 'eslint-plugin-react-hooks';
import refresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'playwright-report/**', 'test-results/**'] },
  js.configs.recommended,
  tseslint.configs.recommended,
  { languageOptions: { globals: { ...globals.browser, ...globals.node } } },
  {
    files: ['src/**/*.{ts,tsx}'],
    plugins: { 'react-hooks': hooks, 'react-refresh': refresh },
    rules: {
      ...hooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['error', { allowConstantExport: true }],
    },
  },
  {
    files: ['src/rendering/**/*.{ts,tsx}'],
    rules: {
      'no-restricted-imports': ['error', {
        patterns: [
          { group: ['react', 'react/**', 'react-dom', 'react-dom/**'], message: 'The renderer is independent from React.' },
          { group: ['**/app/**', '**/components/**'], message: 'The renderer must not depend on application composition.' },
        ],
      }],
      'no-restricted-syntax': ['error', {
        selector: 'JSXElement, JSXFragment', message: 'The renderer must not use JSX or the implicit React runtime.',
      }],
    },
  },
);
