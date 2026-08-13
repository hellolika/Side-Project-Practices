CREATE PROCEDURE [dbo].[HRMS_GetDepartmentManager.1.0.0]
	@MemberId INT
AS
BEGIN
	SET NOCOUNT ON;

	SELECT 
		m.Id,
		m.Username,
		m.Email,
		su.Id as SlackId,
		su.Username as SlackUsername,
		su.RealName as SlackRealName
	FROM Department d 
	LEFT JOIN Team t ON d.Id = t.DepartmentId
	LEFT JOIN Member m ON m.TeamId = t.TeamId
	LEFT JOIN SlackUserInfo su ON su.Email = m.Email
	WHERE m.TeamId = (SELECT TeamId FROM Member WHERE Id = @MemberId) 
	AND m.IsManager = 1
END