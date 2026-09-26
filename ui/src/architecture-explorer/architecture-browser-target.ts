import { createContext, useContext } from 'react';

export const BrowserTarget = createContext<HTMLElement | null | undefined>(undefined);
export const useArchitectureBrowserTarget = () => useContext(BrowserTarget);
