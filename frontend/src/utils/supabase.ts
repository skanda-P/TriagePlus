import { createClient, SupabaseClient } from '@supabase/supabase-js';

let _supabase: SupabaseClient | null = null;

function getSupabase(): SupabaseClient {
  if (!_supabase) {
    const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
    const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
    
    if (!supabaseUrl || !supabaseKey) {
      throw new Error('Supabase URL and Anon Key must be configured in environment variables');
    }
    
    _supabase = createClient(supabaseUrl, supabaseKey, {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    });
  }
  return _supabase;
}

export const supabase = {
  get auth() {
    return getSupabase().auth;
  },
  get from() {
    return getSupabase().from;
  },
  get rpc() {
    return getSupabase().rpc;
  },
  get storage() {
    return getSupabase().storage;
  },
  get functions() {
    return getSupabase().functions;
  },
};