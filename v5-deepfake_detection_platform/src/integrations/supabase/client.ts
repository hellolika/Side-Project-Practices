import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://sqcinwevsucjkisttmst.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNxY2lud2V2c3Vjamtpc3R0bXN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQyMTUyMzEsImV4cCI6MjA3OTc5MTIzMX0.0fXPoF_LcJUpB36Y2wz4lS7aZ9-4Nnusc5gqUfNL-FY'

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Import the supabase client like this:
// For React:
// import { supabase } from "@/integrations/supabase/client";
// For React Native:
// import { supabase } from "@/src/integrations/supabase/client";
