using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class SlackUserResponse
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }
    [JsonProperty("members")]
    public List<SlackUser> Members { get; set; }
}

public class SlackUser
{
    [JsonProperty("name")]
    public string Name { get; set; }
    [JsonProperty("id")]
    public string Id { get; set; }
    [JsonProperty("deleted")]
    public bool Deleted { get; set; }
    [JsonProperty("is_bot")]
    public bool IsBot { get; set; }
    [JsonProperty("real_name")]
    public string RealName { get; set; }
    [JsonProperty("profile")]
    public Profile Profile { get; set; }
}

public class Profile
{
    [JsonProperty("email")]
    public string Email { get; set; }
}

