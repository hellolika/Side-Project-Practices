CREATE PROCEDURE [dbo].[HRMS_GetAllPosition.1.0.0]
AS
BEGIN 
	SET NOCOUNT ON
    SELECT 
        p.[Id], 
        p.[PositionName], 
        p.[TeamId],
        t.[TeamName] 
    FROM [dbo].[Position] p
    LEFT JOIN Team t ON p.TeamId = t.TeamId
END