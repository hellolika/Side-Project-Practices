using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class OneSignalNotificationRequestBase
{
    [JsonProperty("app_id")]
    public string AppId { get; set; }
        
    [JsonProperty("contents")]
    public Dictionary<string,string> Contents { get; set; }
        
    [JsonProperty("headings")]
    public Dictionary<string,string> Headings { get; set; }
        
    [JsonProperty("url")] 
    public string Url { get; set; }

    [JsonProperty("small_icon")]
    public string SmallIcon { get; set; } = "splash";

    [JsonProperty("android_accent_color")]
    public string AndroidAccentColor { get; set; } = "ea3379";
    
    [JsonProperty("channel_for_external_user_ids")]
    public string ChannelExternalIds { get; set; } = "push";

    [JsonProperty("included_segments")]
    public List<string> Segments { get; set; }

    [JsonProperty("include_external_user_ids")]
    public List<string> ExternalIds { get; set; } = new List<string> { };
}